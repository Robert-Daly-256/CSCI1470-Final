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
from april29transformer import ImageCaptionModel, generate_caption
# from keras.models import load_model

def get_image_features(image_path): 
    resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
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
    with open(vocab_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    word2idx = data['word2idx']
    idx2word = data['idx2word']

    return word2idx, idx2word

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"To use: python {sys.argv[0]} <image_path>")
        sys.exit(1)

    # get paths from command line
    img_path = sys.argv[1]
    # model_path = sys.argv[2]
    
    # get image features
    feature, img_array = get_image_features(img_path)
    print("Feature Vector Shape:", feature.shape)

    word2idx, idx2word = load_vocab('data2.p')
    print(len(word2idx))

    caption_model = ImageCaptionModel()
    caption_model.build((1, 2048))
    # caption_model = tf.keras.models.load_model(model_path)
    print("here")

    caption_model.load_weights('icm_apr30.weights.h5')
    # print(model.summary())
    # print(type(model))
    # predicted_caption = caption_model.predict_caption(feature, vocab)
    print("here")
    predicted_caption = generate_caption(caption_model, feature, word2idx, idx2word, 20)
    # greedy_caption = greedy_search(caption_model, feature, word2idx, idx2word, 20)

    print("Generated Caption:", predicted_caption)
    # print("Greedy Capttion:", greedy_caption)

    # show image
    show_image_with_caption_below(img_array, predicted_caption)