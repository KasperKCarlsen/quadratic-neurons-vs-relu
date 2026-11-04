import pickle
import numpy as np
import torch
from functools import lru_cache


def iterate_batches(X, y, batch_size, shuffle=False, drop_last=False):
    """Yield (images, labels) batches directly from device-resident tensors.

    CIFAR-10-scale datasets (a few hundred MB) fit comfortably in a 4070 Ti's
    12 GB of VRAM, so once X/y live on the GPU there is no need to round-trip
    through DataLoader/TensorDataset: collation, pinning and the per-batch
    host->device copy all disappear, leaving just a cheap index_select/slice.
    Shuffling is done once per epoch via a single on-device randperm rather
    than per-batch, to keep kernel-launch count low.
    """
    n = X.size(0)
    order = torch.randperm(n, device=X.device) if shuffle else None
    n_batches = n // batch_size if drop_last else -(-n // batch_size)

    for i in range(n_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, n)
        if order is not None:
            idx = order[start:end]
            yield X.index_select(0, idx), y.index_select(0, idx)
        else:
            yield X[start:end], y[start:end]


def get_classes_dataset(classes, X, y):

    mask = np.isin(y, classes)
    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"Selected classes: {classes}")
    print(f"Total samples: {len(X_filtered)}")
    
    return X_filtered, y_filtered



def load_batch(path, label_key=b'labels'):
    """Load one CIFAR pickle file into (images, labels) numpy arrays.

    label_key selects the label field: CIFAR-10 stores b'labels', CIFAR-100
    stores b'fine_labels' (and b'coarse_labels'). The sample count is inferred
    with -1 rather than hardcoded, so this works for any file size - the
    CIFAR-10 batches (10000 each), the CIFAR-100 train file (50000) and its
    test file (10000) alike.
    """
    with open(path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    images = batch[b'data']        # (N, 3072)
    labels = batch[label_key]      # length N

    images = images.reshape(-1, 3, 32, 32)
    return images, np.array(labels)


@lru_cache(maxsize=8)
def load_checkpoint(path):
    """Load a checkpoint state dict once and reuse it on later calls.

    Quad_adapters/LoRA re-read their pretrained checkpoint from disk every
    time they're constructed, which happens once per Optuna trial (often
    100+ times per run). Caching by path turns that into a single read.
    """
    map_location = "cuda"
    # weights_only=True is safe (checkpoints are plain state_dicts) and silences
    # the PyTorch>=2.4 pickle-security warning / future default flip.
    return torch.load(path, map_location=map_location, weights_only=True)


def freeze_original_weights(model):
    """Freeze original weights, only train LoRA parameters"""
    for name, param in model.named_parameters():
        # For the simple LoRAConv2d implementation
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
        elif 'conv' in name and ('lora' not in name):
            # Freeze original convolution weights
            param.requires_grad = False
        elif 'bn' in name or 'linear' in name:
            # Keep batch norm and final linear layer trainable
            param.requires_grad = True
        else:
            param.requires_grad = False


def resolve_amp_dtype(amp_dtype, device):
    """Map an --amp_dtype string to (use_amp, torch.dtype, needs_grad_scaler).

    bf16 is the A100/Ampere fast path and needs no gradient scaler. V100/Volta
    has no hardware bf16, so 'fp16' must be used there instead - and fp16's
    narrow dynamic range means training wants a GradScaler to avoid gradient
    underflow. 'off' disables autocast entirely (plain fp32).
    """
    if device.type != "cuda" or amp_dtype == "off":
        return False, torch.float32, False
    if amp_dtype == "fp16":
        return True, torch.float16, True
    # default / "bf16"
    return True, torch.bfloat16, False
