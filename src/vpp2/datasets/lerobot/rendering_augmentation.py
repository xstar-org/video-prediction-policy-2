"""Rendering-robustness augmentation for VPP2 training.

Simulates visual differences between 2-pass converged rendering (training data)
and 1-pass / low-resolution rendering (deployment optimization). Applied after
camera concatenation, before resize/crop/normalize.
"""
import random

import torch
import torch.nn.functional as F


class RenderingRobustnessAugmentation:
    """Randomly degrade video to build tolerance to rendering changes.

    Three degradation modes (applied independently with probability p_mode):
    1. Resolution drop: downscale to random resolution, then upscale back
       (simulates low-res wrists or reduced render quality)
    2. Appearance jitter: brightness/contrast shift + Gaussian noise
       (simulates 1-pass temporal convergence differences)
    3. Combined: both at once

    The model sees clean (original) data with probability 1 - p_any,
    ensuring it doesn't forget the canonical rendering distribution.
    """

    def __init__(
        self,
        p_any: float = 0.4,
        p_resolution: float = 0.5,
        p_appearance: float = 0.5,
        min_scale: float = 0.25,
        brightness_range: float = 0.08,
        contrast_range: float = 0.08,
        noise_std: float = 0.03,
    ):
        self.p_any = p_any
        self.p_resolution = p_resolution
        self.p_appearance = p_appearance
        self.min_scale = min_scale
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a video tensor.

        Args:
            video: (C, T, H, W) or (T, C, H, W) in [0, 1] range.

        Returns:
            Augmented video, same shape, still in [0, 1].
        """
        if random.random() > self.p_any:
            return video  # keep clean

        # Ensure (C, T, H, W) internally
        if video.dim() == 4 and video.shape[0] != 3:
            # (T, C, H, W) -> (C, T, H, W)
            was_tchw = True
            video = video.permute(1, 0, 2, 3)
        else:
            was_tchw = False

        C, T, H, W = video.shape

        do_resolution = random.random() < self.p_resolution
        do_appearance = random.random() < self.p_appearance
        if not do_resolution and not do_appearance:
            # Guarantee at least one degradation when p_any triggered
            do_resolution = True

        if do_resolution:
            video = self._resolution_drop(video)

        if do_appearance:
            video = self._appearance_jitter(video)

        video = video.clamp(0, 1)

        if was_tchw:
            video = video.permute(1, 0, 2, 3)
        return video

    def _resolution_drop(self, video: torch.Tensor) -> torch.Tensor:
        """Downscale to random resolution, then upscale back (bicubic)."""
        C, T, H, W = video.shape
        scale = random.uniform(self.min_scale, 1.0)
        h_low = max(8, int(H * scale))
        w_low = max(8, int(W * scale))
        # (C, T, H, W) -> (C*T, 1, H, W) for F.interpolate
        flat = video.reshape(C * T, 1, H, W)
        flat = F.interpolate(flat, size=(h_low, w_low), mode="bicubic", align_corners=False, antialias=True)
        flat = F.interpolate(flat, size=(H, W), mode="bicubic", align_corners=False, antialias=True)
        return flat.reshape(C, T, H, W)

    def _appearance_jitter(self, video: torch.Tensor) -> torch.Tensor:
        """Apply random brightness/contrast shift + Gaussian noise."""
        brightness = random.uniform(-self.brightness_range, self.brightness_range)
        contrast_center = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
        video = (video - 0.5) * contrast_center + 0.5 + brightness
        if self.noise_std > 0:
            noise = torch.randn_like(video) * self.noise_std
            video = video + noise
        return video
