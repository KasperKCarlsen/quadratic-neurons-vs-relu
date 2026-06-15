import pickle
import numpy as np


def get_classes_dataset(classes, X, y):

    mask = ""
    for i in classes:
        mask = mask + f"(y == {i}) | "

    mask = mask[:-3]
    mask = eval(mask)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"Selected classes: {classes}")
    print(f"Total samples: {len(X_filtered)}")
    
    return X_filtered, y_filtered



def load_batch(path, label_key=b'labels'):
    """Load a CIFAR pickle. label_key selects the label field: b'labels' for
    CIFAR-10, b'fine_labels' for CIFAR-100 (100 classes). reshape(-1, ...)
    handles both the 10k-image CIFAR-10 batches and the 50k-image CIFAR-100
    train file."""
    with open(path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data']
    labels = batch[label_key]
    images = images.reshape(-1, 3, 32, 32)
    return images, np.array(labels)

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


# For datasets package
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
import torch


def extract_features(dataset_split):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    feature_extractor.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)
    
    features = []
    labels = []
    
    for i in tqdm(range(len(dataset_split)), desc="Extracting features"):
        image = dataset_split[i]["img"].convert("RGB")
        label = dataset_split[i]["label"]
        
        # Preprocess
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features (use pooler_output)
        with torch.no_grad():
            outputs = feature_extractor(**inputs)
            feature = outputs.pooler_output.cpu().numpy()  # [768]
        
        features.append(feature.squeeze())
        labels.append(label)
    
    return np.array(features), np.array(labels)