"""Request-local workspace for a frozen video prefix and changing action suffix."""
import torch


def join_action_kv(cache, k_video, v_video, k_action, v_action):
    """Copy each frozen prefix once; preserve ordinary cat for unsupported cases.

    The cache owns its prefix views and workspaces. It must be private to one
    serial inference request; the caller keeps training and routed mixed-prefix
    paths outside this helper. No model state or cross-request cache is retained.
    """
    values=(k_video,v_video,k_action,v_action)
    eligible=(not torch.is_grad_enabled()
              and cache.get('k') is k_video and cache.get('v') is v_video
              and all(t.ndim==3 and t.dtype==k_video.dtype and t.device==k_video.device for t in values)
              and all(t.shape[0]==k_video.shape[0] and t.shape[2]==k_video.shape[2] for t in values)
              and k_video.shape==v_video.shape and k_action.shape==v_action.shape)
    if not eligible:
        return torch.cat([k_video,k_action],dim=1),torch.cat([v_video,v_action],dim=1)
    prefix=int(k_video.shape[1])
    shape=(k_video.shape[0],prefix+k_action.shape[1],k_video.shape[2])
    joined=[]
    for key,video,action in [('k',k_video,k_action),('v',v_video,v_action)]:
        slot='_action_joint_'+key
        buf=cache.get(slot)
        if buf is None or tuple(buf.shape)!=shape or buf.dtype!=video.dtype or buf.device!=video.device:
            buf=video.new_empty(shape)
        view=buf[:,:prefix]
        # Prefix views alias the workspace; writing the suffix on this same
        # stream cannot change them. A replaced source prefix is copied again.
        if video.data_ptr()!=view.data_ptr() or video.stride()!=view.stride():
            view.copy_(video)
        buf[:,prefix:].copy_(action)
        cache[key]=view
        cache[slot]=buf
        joined.append(buf)
    return tuple(joined)
