from contextlib import contextmanager

import torch


@contextmanager
def autocast_for_device(device, enabled=True):
    enabled = bool(enabled)
    device_type = getattr(device, "type", str(device))
    if not enabled:
        yield
        return
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast(device_type=device_type, enabled=enabled):
            yield
        return
    if device_type == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
        with torch.cuda.amp.autocast(enabled=enabled):
            yield
        return
    yield


def make_grad_scaler(enabled=True):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    return _NullGradScaler()


class _NullGradScaler(object):
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def inference_context():
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()
