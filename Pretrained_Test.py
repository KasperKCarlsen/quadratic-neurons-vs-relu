# #%%
# from transformers import ViTImageProcessor, ViTModel
# from PIL import Image
# import requests


# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
# model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')


# #%%
# from torchvision.datasets import CIFAR10

# dataset = CIFAR10(
#     root="./data",
#     train=False,
#     download=True
# )

# # image, label = dataset[0]

# # inputs = processor(
# #     images=image,
# #     return_tensors="pt"
# # )

# # outputs = model(**inputs)
# # print(outputs)



#%%
from transformers import ViTModel, ViTImageProcessor
from datasets import load_dataset
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

# Load model and processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
feature_extractor.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)

# Load CIFAR-10
dataset = load_dataset("cifar10")

# Extract features from all test images
def extract_features(dataset_split):
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

# Extract training and test features
print("Extracting training features...")
X_train, y_train = extract_features(dataset["train"])
print("Extracting test features...")
X_test, y_test = extract_features(dataset["test"])

# Train a classifier on the features
classifier = LogisticRegression(max_iter=1000, C=1.0)
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (zero-shot features + logistic regression): {accuracy:.4f}")