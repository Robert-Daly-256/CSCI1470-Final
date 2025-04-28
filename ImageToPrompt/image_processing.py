import pickle
import random
import re
from PIL import Image
import tensorflow as tf 
import numpy as np
import collections
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

def get_image_features(image_path): 
    resnet = tf.keras.applications.ResNet50(include_top=False)
    gap = tf.keras.layers.GlobalAveragePooling2D()

    with Image.open(image_path) as img:
        img_array = np.array(img.resize((224, 224)))
    img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]

    features = gap(resnet(img_in))
    return features, img_array

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"To use: python {sys.argv[0]} <image_path>")
        sys.exit(1)

    # get image features
    img_path = sys.argv[1]
    feature, img_array = get_image_features(img_path)
    print("Feature Vector Shape:", feature.shape)

    # show image
    plt.imshow(img_array.astype('uint8'))  
    plt.axis('off')  
    plt.title('Input Image')
    plt.show()