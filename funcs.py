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



def load_batch(path):
    with open(path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        
    images = batch[b'data']      # (10000, 3072)
    labels = batch[b'labels']    # length 10000

    images = images.reshape(10000, 3, 32, 32)
    return images, np.array(labels)


@lru_cache(maxsize=8)
def load_checkpoint(path):
    """Load a checkpoint state dict once and reuse it on later calls.

    Quad_adapters/LoRA re-read their pretrained checkpoint from disk every
    time they're constructed, which happens once per Optuna trial (often
    100+ times per run). Caching by path turns that into a single read.
    """
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.load(path, map_location=map_location)


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
