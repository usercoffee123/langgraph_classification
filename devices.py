"""Select the accelerator shared by the local detectors."""


def resolve_device(requested: str = 'auto') -> str:
    import torch

    if requested not in {'auto', 'mps', 'cuda', 'cpu'}:
        raise ValueError('Device must be auto, mps, cuda, or cpu.')
    if requested == 'auto':
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable. Run with native Apple Silicon Python and an MPS-enabled PyTorch, or use --device cpu.')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device auto or cpu.')
    return requested
