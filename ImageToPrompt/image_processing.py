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
# from keras.models import load_model

def get_image_features(image_path): 
    resnet = tf.keras.applications.ResNet50(include_top=False)
    gap = tf.keras.layers.GlobalAveragePooling2D()

    with Image.open(image_path) as img:
        img_array = np.array(img.resize((224, 224)))
    img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]

    features = gap(resnet(img_in))
    return features, img_array

def show_image_with_caption_below(img_array, caption):
    fig, ax = plt.subplots()
    ax.imshow(img_array.astype('uint8'))
    ax.axis('off')

    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

def load_vocab(vocab_path):
     # load vocab
    with open('/Users/annieherring/Documents/s2025/dl/CSCI1470-Final/ImageToPrompt/image_data.p', 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    vocab = data['word2idx']
    return vocab

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"To use: python {sys.argv[0]} <image_path> <model_path>")
        sys.exit(1)

    # get paths from command line
    img_path = sys.argv[1]
    model_path = sys.argv[2]
    
    # get image features
    feature, img_array = get_image_features(img_path)
    print("Feature Vector Shape:", feature.shape)

    vocab = load_vocab('/Users/annieherring/Documents/s2025/dl/CSCI1470-Final/ImageToPrompt/image_data.p')

    model = tf.keras.models.load_model(model_path)
    # print(model.summary())
    # print(type(model))
    predicted_caption = model.predict_caption(feature, vocab)

    print("Generated Caption:", predicted_caption)

    # show image
    # show_image_with_caption_below(img_array, predicted_caption)