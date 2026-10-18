"""Versioned native-frame protocol for the released RoboDojo history schedule.

Frames retain their absolute simulator action index. Skipped frames are never
filled with a copy of a nearby image. Dense recording remains the default.
"""
from collections import deque

NATIVE_STEP_FIELD = "_vpp2_history_native_step"
PROTOCOL_VERSION = 1


def native_step(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Native step must be a nonnegative integer")
    return value


def needs_frame(step):
    step = native_step(step)
    return any((step + offset * 25) % 24 == 0 for offset in range(8))


def check_configuration(policy):
    if (policy.history_stride != 25 or policy.history_condition_frames != 8
            or policy.replan_steps != 24):
        raise ValueError("Native-frame v1 requires replan24, anchor+history8, stride25")


def record_frame(policy, image, env, step, convert):
    check_configuration(policy)
    step = native_step(step)
    if not needs_frame(step):
        raise ValueError("Observation lies outside the negotiated native-frame schedule")
    last = policy._history_step_by_env.get(env)
    if ((last is None and step != 0) or (last is not None and step <= last)):
        raise ValueError("Missing anchor, duplicate, or out-of-order native frame")
    frame = convert(image)
    if last is None:
        policy._history_by_env[env] = deque(maxlen=policy.history_buffer_size)
        policy._history_anchor_by_env[env] = frame
    policy._history_by_env[env].append((step, frame))
    policy._history_step_by_env[env] = step


def validate_inference(policy, observation, env):
    check_configuration(policy)
    step = native_step(observation[NATIVE_STEP_FIELD])
    expected = policy._inference_index_by_env.get(env, 0) * 24
    if step != expected or policy._history_step_by_env.get(env) != step:
        raise ValueError("Inference requires its genuine current native frame")
    recorded = dict(policy._history_by_env[env])
    for offset in range(8):
        required = step - offset * 25
        if required > 0 and required not in recorded:
            raise ValueError(f"Missing required native history frame {required}")
