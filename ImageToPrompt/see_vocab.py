import pickle
import tensorflow as tf

# load vocab
with open('/Users/annieherring/Documents/s2025/dl/CSCI1470-Final/ImageToPrompt/image_data.p', 'rb') as f:
    data = pickle.load(f)

# check vocab (word2idx) to see if certain tokens are included
vocab = data['word2idx']
print("First 10 entries in vocab:", list(vocab.items())[:10])
print("Does '<start>' exist in vocab?", '<start>' in vocab)
print("Does '<end>' exist in vocab?", '<end>' in vocab)