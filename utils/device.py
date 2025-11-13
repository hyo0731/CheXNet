import torch

# --- Device 설정 (CUDA/MPS/CPU) ---
def get_device():
    """Determines the appropriate device (MPS, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")