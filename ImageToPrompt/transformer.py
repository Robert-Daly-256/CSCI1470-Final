import math
import numpy as np
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="transformer_layers")
class TransformerBlock(keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        # 1) Define the Feed Forward, self-attention, encoder-decoder-attention, and layer normalization layers
        # 2) Use multiheaded attention if multiheaded is True!

        self.multiheaded = multiheaded
        self.emb_sz = emb_sz

        self.feed_forward = tf.keras.layers.Dense(emb_sz, activation='relu')

        # if multiheaded:
        #     self.self_attention = MultiHeadedAttention(emb_sz, use_mask=True)
        #     self.ed_attention = MultiHeadedAttention(emb_sz, use_mask=False)
        # else:
        #     self.self_attention = AttentionHead(emb_sz, emb_sz, is_self_attention=True)
        #     self.ed_attention = AttentionHead(emb_sz, emb_sz, is_self_attention=False)
        self.self_att = keras.layers.MultiHeadAttention(num_heads=3, key_dim=emb_sz, dropout=0.1)
        self.unmasked_att = keras.layers.MultiHeadAttention(num_heads=3, key_dim=emb_sz, dropout=0.1)

        self.ffn = keras.Sequential(
            [keras.layers.Dense(emb_sz, activation="relu"), keras.layers.Dense(emb_sz),]
        )
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
    

    def call(self, inputs, context_sequence, training=False):
        """
        This functions calls a transformer block.

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """

        # TODO:
        # 1) compute MASKED attention on the inputs 
        # 2) residual connection and layer normalization 
        attn_mask = None
        self_attn_output = self.self_att(inputs, inputs, inputs, attention_mask=attn_mask, training=training)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.norm1(inputs + self_attn_output)

        # 3) computed UNMASKED attention using context
        # 4) residual connection and layer normalization
        # attn_output = self.unmasked_att(context_sequence, context_sequence, out1)
        # attn_output = self.dropout(attn_output)
        # out2 = self.norm2(out1 + attn_output)
        unmasked_attn_output = self.unmasked_att(out1, context_sequence, context_sequence, training=training)
        unmasked_attn_output = self.dropout2(unmasked_attn_output, training=training)
        out2 = self.norm2(out1 + unmasked_attn_output)

        # 5) feed forward layer
        # 6) residual layer and layer normalization
        ff_output = self.feed_forward(out2)
        ff_output = self.dropout2(ff_output)
        out3 = self.norm3(out2 + ff_output)

        # 7) call relu and return tensor
        return tf.nn.relu(out3)

@keras.saving.register_keras_serializable(package="transformer_layers", name="positional_encoding")
def positional_encoding(length, depth):
    ## TODO:
    depth = depth // 2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)


@keras.saving.register_keras_serializable(package="transformer_layers")
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)

        ## Sinosoidal positional encoding: offset by varying sinosoidal frequencies.
        ## HINT: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
        self.pos_encoding = positional_encoding(length=window_size, depth=embed_size)[..., :window_size, :]

    def call(self, x):
        ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
        len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :len, :]

        return x
