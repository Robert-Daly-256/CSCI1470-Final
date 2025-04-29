import tensorflow as tf
import keras
try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

@keras.saving.register_keras_serializable(package="MyLayers")
class TransformerDecoder(keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # Define feed forward layer to embed image features into a vector 
        # with the models hidden size
        self.image_embedding = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.english_embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)

        # learnable positional encoding
        self.positional_embeddings = self.add_weight(
            name="positional_embeddings",
            shape=(window_size, hidden_size),
            initializer="random_normal",
        )
        self.dropout = tf.keras.layers.Dropout(0.1)

        # Define positional encoding layer for language:
        # self.positional_encoding = PositionalEncoding(vocab_size, hidden_size, window_size)
        self.simple_encoding = SimpleEncoder(hidden_size)

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

        # 3) Add positional embeddings to the word embeddings
        english_embeddings = self.english_embedding(captions)

        # with simple encoder
        english_embeddings = self.simple_encoding(english_embeddings)
        decoder_output = self.transformer_decoder(english_embeddings, image_features)
        
        # with learnable positional encoding
        # positions = tf.range(start=0, limit=self.window_size, delta=1)
        # pos_embeds = tf.nn.embedding_lookup(self.positional_embeddings, positions)
        # word_embeddings = english_embeddings + pos_embeds
        # x = self.dropout(word_embeddings, training=True)
        # decoder_output = self.transformer_decoder(x, image_features)

        # with both learnable positional encoder and simple encoder
        # positions = tf.range(start=0, limit=self.window_size, delta=1)
        # pos_embeds = tf.nn.embedding_lookup(self.positional_embeddings, positions)
        # word_embeddings = english_embeddings + pos_embeds  
        # word_embeddings = self.simple_encoding(word_embeddings)
        # decoder_output = self.transformer_decoder(word_embeddings, image_features)

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

@keras.saving.register_keras_serializable(package="MyLayers")
class SimpleEncoder(keras.layers.Layer):
    def __init__(self, hidden_size, num_heads=4, ff_dim=256):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size)
        self.feedforward = keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(hidden_size),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn_output = self.attention(x, x)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.feedforward(x)
        return self.layernorm2(x + ffn_output)
