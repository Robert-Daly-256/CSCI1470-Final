import tensorflow as tf
import keras
try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

@keras.saving.register_keras_serializable(package="MyLayers")
class RNNDecoder(keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Now we will define image and word embedding, decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        # with the models hidden size
        self.image_embedding = keras.layers.Dense(hidden_size, activation="relu")
        self.img_embedding_seq = keras.Sequential()

        # Define english embedding layer:
        self.english_embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden_size, mask_zero=True)

        # Define decoder layer:     
        self.decoder_rnn = keras.layers.LSTM(hidden_size, return_sequences=True)

        self.dropout = keras.layers.Dropout(0.1)
        
        # Define classification layer:
        self.classifier = keras.layers.Dense(vocab_size)
        self.classifier_seq = keras.Sequential() 
        
    def call(self, encoded_images, captions):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions: tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
        """

        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension
        image_embeddings = self.image_embedding(encoded_images)
        # image_embeddings = self.img_embedding_seq(image_embeddings)

        # cap_shift = english_embeddings[:, :-1] # remove the first word (teacher forcing)
        english_embeddings = self.english_embedding(captions)
        
        # 2) Pass your english sentence embeddings, and the image embeddings, to your decoder 
        decoder_output = self.decoder_rnn(english_embeddings, initial_state=[image_embeddings, image_embeddings])
        decoder_output = self.dropout(decoder_output)

        # 3) Apply dense layer(s) to the decoder out to generate logits
        logits = self.classifier(decoder_output)
        # logits = self.classifier_seq(decoder_output)

        return logits

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "window_size"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

########################################################################################

@keras.saving.register_keras_serializable(package="MyLayers")
class TransformerDecoder(keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Now we will define image and word embedding, positional encoding, tramnsformer decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        # with the models hidden size
        self.image_embedding = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.english_embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)

        # Define positional encoding layer for language:
        self.positional_encoding = PositionalEncoding(vocab_size, hidden_size, window_size)

        # Define transformer decoder layer:
        self.transformer_decoder = TransformerBlock(hidden_size, multiheaded=True)

        # Define classification layer
        self.classification_layer = tf.keras.layers.Dense(vocab_size)
        

    def call(self, encoded_images, captions):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions: tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
        """
        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension
        image_features = self.image_embedding(encoded_images)
        image_features = tf.expand_dims(image_features, axis=1)
        
        # 2) Pass the captions through your word embedding layer
        # english_embeddings = self.english_embedding(captions)

        # 3) Add positional embeddings to the word embeddings
        english_embeddings = self.positional_encoding(captions)
        
        # 4) Pass the english embeddings and the image sequences, to the decoder
        decoder_output = self.transformer_decoder(english_embeddings, image_features)
        
        # 5) Apply dense layer(s) to the decoder out to generate **logits**
        logits = self.classification_layer(decoder_output)
        
        return logits

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "window_size"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)    
