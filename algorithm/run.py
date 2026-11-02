"""
Evaluation runs (no gradients). Each returns

    kkt       : list of KKT residuals, one per iteration
    x_hist    : list of primal iterates [u, w], one per iteration
"""

import torch

from algorithm.fbs_step import unroll


def _floats(kkt, x_hist):
    return [t.item() for t in kkt], x_hist


def run_zero(initial_state, functions, params, shapes, T, device):
    """Baseline: no deviations."""
    with torch.no_grad():
        return _floats(*unroll(functions, params, shapes, T, device,
                               keep_history=True, progress=True))


def run_random(initial_state, functions, params, shapes, T=100, device="cuda",
               alpha=0.99, seed=0):
    """Control: random directions, rescaled by the safeguard."""
    torch.manual_seed(seed)

    def direction(x, *_):
        return [torch.randn_like(b) for b in x], [torch.randn_like(b) for b in x]

    with torch.no_grad():
        return _floats(*unroll(functions, params, shapes, T, device,
                               direction=direction, alpha=alpha,
                               keep_history=True, progress=True))


def run_learned(model, initial_state, clean, functions, T_test=500, return_all=False):
    """Learned deviations (safeguarded or not, as the model was trained).
    With return_all, also returns {"x": x_hist}."""
    model.eval()
    with torch.no_grad():
        kkt, x_hist = _floats(*model(
            functions, T=T_test,
            keep_history=return_all, progress=True))
    if return_all:
        return kkt, {"x": x_hist}
    return kkt
