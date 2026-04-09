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



def load_batch(path):
    with open(path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        
    images = batch[b'data']      # (10000, 3072)
    labels = batch[b'labels']    # length 10000

    images = images.reshape(10000, 3, 32, 32)
    return images, np.array(labels)