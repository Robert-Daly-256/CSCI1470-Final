import pickle
import numpy as np

with open("./data/data.p", 'rb') as data_file:
    data_dict = pickle.load(data_file)

feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 2048), 5, axis=0)
img_prep  = lambda x: np.repeat(x, 5, axis=0)
train_captions  = np.array(data_dict['train_captions'])
test_captions   = np.array(data_dict['test_captions'])
train_img_feats = feat_prep(data_dict['train_image_features'])
test_img_feats  = feat_prep(data_dict['test_image_features'])
train_images    = img_prep(data_dict['train_images'])
test_images     = img_prep(data_dict['test_images'])
word2idx        = data_dict['word2idx']
idx2word        = data_dict['idx2word']