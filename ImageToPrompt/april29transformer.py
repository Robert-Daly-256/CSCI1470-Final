import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn1 = layers.Dense(dff, activation='relu')
        self.ffn2 = layers.Dense(d_model)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=True): 
        # please work 
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # final feed forward layers
        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)

        # output normalized
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.ffn1 = layers.Dense(dff, activation='relu')
        self.ffn2 = layers.Dense(d_model)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training=True, look_ahead_mask=None):
        # call transformer decoder layer 
        attn1 = self.mha1(query=x, value=x, key=x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(query=out1, value=enc_output, key=enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # ff layers
        ffn_output = self.ffn1(out2)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)

        # final normalization 
        return self.layernorm3(ffn_output + out2)

def create_padding_mask(seq):
    # masks for padding tokens
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    # block off ability to look ahead
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # make masks for attention
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask, dec_padding_mask

def positional_encoding(position, d_model):
    # pos encoding for transformer -- same scaffolding as hw4
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
        
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

class ImageCaptionModel(Model):
    def __init__(self, 
                 vocab_size=639,
                 d_model=256, 
                 num_heads=8, 
                 dff=1024,
                 max_position_encoding=20,
                 num_layers=4,
                 dropout_rate=0.1):
        super(ImageCaptionModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)
        
        # image encoding
        self.image_feature_extractor = layers.Dense(d_model)
        self.image_feature_dropout = layers.Dropout(dropout_rate)
        
        # encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate) 
            for _ in range(num_layers)
        ]
        
        # decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate) 
            for _ in range(num_layers)
        ]
        
        # output layer
        self.final_layer = layers.Dense(vocab_size)
    
    def encode_image(self, img_features, training):
        img_enc = self.image_feature_extractor(img_features)
        img_enc = self.image_feature_dropout(img_enc, training=training)
        img_enc = tf.expand_dims(img_enc, axis=1)

        for encoder_layer in self.encoder_layers:
            img_enc = encoder_layer(img_enc, training=training)

        return img_enc
    
    def decode_sequence(self, x, enc_output, training, look_ahead_mask=None):
        seq_len = tf.shape(x)[1]
        
        # get embeddings
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        # loop through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, training=training, look_ahead_mask=look_ahead_mask)
            
        # linear
        output = self.final_layer(x)
        
        return output
        
    def call(self, inputs, training=True):
        img_features, captions = inputs
        
        # make mask
        look_ahead_mask, _ = create_masks(img_features, captions)
        
        # use encoder
        enc_output = self.encode_image(img_features, training)
        
        # use decoder
        dec_output = self.decode_sequence(
            captions, enc_output, training, look_ahead_mask)
        
        return dec_output

def compile_model(model, args):
    # compiles model
    optimizer = optimizers.Adam(
        learning_rate=args.learning_rate, 
        beta_1=0.9, 
        beta_2=0.98, 
        epsilon=1e-9
    )
    
    # loss function 
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
        
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    
    model.compile(optimizer=optimizer, loss=loss_function)
    return model

def train_model(model, train_captions, train_img_feats, pad_token, args, valid=None):
    # trains model on input data

    # teacher forcing
    decoder_input = train_captions[:, :-1]  # rm last token
    decoder_target = train_captions[:, 1:]  # rm first token
    
    # Create validation data if provided
    validation_data = None
    if valid is not None:
        valid_captions, valid_img_feats = valid
        valid_decoder_input = valid_captions[:, :-1]
        valid_decoder_target = valid_captions[:, 1:]
        validation_data = (
            [valid_img_feats, valid_decoder_input],
            valid_decoder_target
        )
    
    callbacks = []
    
    # checkpoint path added in 
    if args.chkpt_path:
        checkpoint_callback = ModelCheckpoint(
            filepath=args.chkpt_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss' if valid is not None else 'loss',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
    
    # early stopping
    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss' if valid is not None else 'loss',
            patience=args.patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
    
    # train model 
    history = model.fit(
        [train_img_feats, decoder_input],
        decoder_target,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=validation_data,
        callbacks=callbacks
    )
    
    return history

def save_model(model, args):
    # saves the model to the input path
    if args.save_format == 'h5':
        model.save_weights(args.chkpt_path)
    else:
        model.save(args.chkpt_path, save_format=args.save_format)

# ----- make caption -----
def generate_caption(model, image_features, word2idx, idx2word, max_length=50):
    # generates caption for an image 
    start_token = word2idx.get('<start>', word2idx.get('<sos>', 1))
    end_token = word2idx.get('<end>', word2idx.get('<eos>', 2))
    
    # start with <start>
    decoder_input = tf.expand_dims([start_token], 0)

    caption = []
    
    # encode
    enc_output = model.encode_image(tf.expand_dims(image_features, 0), training=False)
    
    for i in range(max_length):
        # get a prediction!
        predictions = model.decode_sequence(
            decoder_input, 
            enc_output, 
            training=False
        )
        
        # last predicted token 
        prediction_id = tf.argmax(predictions[:, -1:, :], axis=-1)
        prediction_id = prediction_id.numpy()[0][0]
        
        # stop if the sentence is over
        if prediction_id == end_token:
            break
            
        # add to result
        caption.append(idx2word[prediction_id])
        
        # update decoder
        decoder_input = tf.concat([decoder_input, tf.expand_dims([prediction_id], 0)], axis=-1)
    
    return ' '.join(caption)

# parse cmnd line 
class Args:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 64
        self.epochs = 20
        self.early_stopping = True
        self.patience = 5
        self.chkpt_path = 'image_caption_model.weights.h5'
        self.save_format = 'h5'

if __name__ == "__main__":
    args = Args()

    with open('/Users/annieherring/Documents/s2025/dl/CSCI1470-Final/ImageToPrompt/data2.p', 'rb') as data_file:
        data_dict = pickle.load(data_file)

    feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 2048), 5, axis=0)
    # img_prep  = lambda x: np.repeat(x, 5, axis=0)
    train_captions  = np.array(data_dict['train_captions'])
    test_captions   = np.array(data_dict['test_captions'])
    
    train_img_feats = feat_prep(data_dict['train_image_features'])
    test_img_feats  = feat_prep(data_dict['test_image_features'])

    word2idx        = data_dict['word2idx']

    model = ImageCaptionModel()
    
    compile_model(model, args)
    
    train_model(
        model, train_captions, train_img_feats, word2idx['<pad>'], args, 
        valid=(test_captions, test_img_feats)
    )
    
    # save model
    if args.chkpt_path:
        save_model(model, args)