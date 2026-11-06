"""Scoped backward memory controls; never installed on forward-only tool paths."""
from contextlib import contextmanager
from functools import wraps


PEAK_MEMORY = 'cuda_peak_memory_allocated_bytes'


def peak_memory(function):
    """CUDA-only receipt; statistics retain their list API (receipt on each row)."""
    @wraps(function)
    def measured(student, *args, **kwargs):
        import torch
        cuda = student.device.type == 'cuda'
        if cuda:
            torch.cuda.reset_peak_memory_stats(student.device)
        result = function(student, *args, **kwargs)
        if cuda:
            peak = torch.cuda.max_memory_allocated(student.device)
            for row in result if isinstance(result, list) else [result]:
                row[PEAK_MEMORY] = peak
        return result
    return measured


def _checkpointed(forward):
    @wraps(forward)
    def wrapped(*args, **kwargs):
        import torch
        from torch.utils.checkpoint import checkpoint
        # E.g. fresh_thought generation inside a backward tool still runs natively.
        if not torch.is_grad_enabled():
            return forward(*args, **kwargs)
        # Gemma4 publishes shared KV through a mutable dictionary. Replay must see
        # the entry-time state and must not replace the live forward's KV tensors.
        shared = kwargs.get('shared_kv_states')
        before = dict(shared) if shared is not None else None
        replay = False

        def run(*inputs, **options):
            nonlocal replay
            if replay and before is not None:
                options['shared_kv_states'] = before.copy()
            replay = True
            return forward(*inputs, **options)

        return checkpoint(run, *args, use_reentrant=False, **kwargs)
    return wrapped


@contextmanager
def checkpoint_layers(layers):
    """Keep eval/dropout settings, objects, hooks and instance-forward identity intact."""
    originals = []
    try:
        for layer in layers:
            had_forward = 'forward' in layer.__dict__
            original = layer.__dict__.get('forward')
            originals.append((layer, had_forward, original))
            layer.forward = _checkpointed(layer.forward)
        yield
    finally:
        for layer, had_forward, original in reversed(originals):
            if had_forward:
                layer.forward = original
            else:
                del layer.forward
