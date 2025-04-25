import math
import numpy as np
import tensorflow as tf
import keras
@keras.saving.register_keras_serializable(package="transformer_layers")
class AttentionMatrix(keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """
        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys    = K.get_shape()[1]  # window size of keys
        embedding_size_keys = K.get_shape()[2]

        mask = tf.convert_to_tensor(
            value=np.transpose(np.tril(np.ones((window_size_queries, window_size_keys)) * np.NINF, -1), (1, 0)),
            dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        # TODO:
        # 1) compute attention weights using queries and key matrices (if use_mask==True, then make sure to add the attention mask before softmax)
        multiplied_qk = tf.matmul(Q, K, transpose_b = True)
        dk = tf.cast(embedding_size_keys, tf.float32)
        scaled_attention_logits = multiplied_qk / tf.sqrt(dk)

        # add in mask if use_mask = true
        if self.use_mask:
            scaled_attention_logits += atten_mask  

        # softmax 
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # attention_weights = tf.matmul(attention_weights, K)

        # 2) return the attention matrix 
        return attention_weights

        # Check lecture slides for how to compute self-attention
        # Remember:
        # - Q is [batch_size x window_size_queries x embedding_size]
        # - K is [batch_size x window_size_keys x embedding_size]
        # - Mask is [batch_size x window_size_queries x window_size_keys]

        # Here, queries are matmuled with the transpose of keys to produce for every query vector, weights per key vector.
        # This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
        # Those weights are then used to create linear combinations of the corresponding values for each query.
        # Those queries will become the new embeddings.

@keras.saving.register_keras_serializable(package="transformer_layers")
class AttentionHead(keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        # TODO:
        # Initialize the weight matrices for K, V, and Q.
        # They should be able to produce a (batch_size, output_size) tensor
        # Hint: use self.add_weight(...) - refer to the handout for more information!

        # QUESTION: how do we get batch_size here? 
        self.weights_Q = self.add_weight(name="Q", shape=(input_size, output_size), initializer="glorot_uniform", trainable=True)
        self.weights_K = self.add_weight(name="K", shape=(input_size, output_size), initializer="glorot_uniform", trainable=True)
        self.weights_V = self.add_weight(name="V", shape=(input_size, output_size), initializer="glorot_uniform", trainable=True)
        
        # Initialize the AttentionMatrix layer
        self.attention_matrix = AttentionMatrix(use_mask=self.use_mask)
        

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        # TODO:
        # - Apply 3 matrices to turn inputs into keys, values, and queries. You will need to use tf.tensordot for this.
        K = tf.tensordot(inputs_for_keys, self.weights_K, axes=[[2], [0]])  # [batch_size, key_window_size, output_size]
        V = tf.tensordot(inputs_for_values, self.weights_V, axes=[[2], [0]])  # [batch_size, key_window_size, output_size]
        Q = tf.tensordot(inputs_for_queries, self.weights_Q, axes=[[2], [0]])  # [batch_size, query_window_size, output_size]

        # - Call AttentionMatrix with the keys and queries.
        attn_weights = self.attention_matrix([K, Q])

        # - Apply the attention matrix to the values.
        attn_output = tf.matmul(attn_weights, V)

        return attn_output

@keras.saving.register_keras_serializable(package="transformer_layers")
class MultiHeadedAttention(keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)

        # Initialize Attention Heads Here
        self.emb_sz = emb_sz
        self.use_mask = use_mask
        self.num_heads = 3
        self.head_dim = emb_sz // self.num_heads

        # self.weights_Q = self.add_weight(shape=(emb_sz, self.head_dim), initializer="glorot_uniform", trainable=True)
        # self.weights_K = self.add_weight(shape=(emb_sz, self.head_dim), initializer="glorot_uniform", trainable=True)
        # self.weights_V = self.add_weight(shape=(emb_sz, self.head_dim), initializer="glorot_uniform", trainable=True)
        
        self.attn1 = AttentionHead(emb_sz, self.head_dim, is_self_attention=use_mask)
        self.attn2 = AttentionHead(emb_sz, self.head_dim, is_self_attention=use_mask)
        self.attn3 = AttentionHead(emb_sz, self.head_dim, is_self_attention=use_mask)

        self.linear_layer = tf.keras.layers.Dense(self.emb_sz, activation='linear')

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a multiheaded attention layer.

        Requirements:
            - Splits data for 3 different heads of size embed_sz/3
            - Create three different attention heads
            - Concatenate the outputs of these heads together
            - Apply a linear layer

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        # K = tf.tensordot(inputs_for_keys, self.weights_K, axes=1)  # [batch_size, key_window_size, output_size]
        # V = tf.tensordot(inputs_for_values, self.weights_V, axes=1)  # [batch_size, key_window_size, output_size]
        # Q = tf.tensordot(inputs_for_queries, self.weights_Q, axes=1)  # [batch_size, query_window_size, output_size]

        attn1 = self.attn1(inputs_for_keys, inputs_for_values, inputs_for_queries)
        attn2 = self.attn2(inputs_for_keys, inputs_for_values, inputs_for_queries)
        attn3 = self.attn3(inputs_for_keys, inputs_for_values, inputs_for_queries)
        
        concatenated = tf.concat([attn1, attn2, attn3], axis=-1)
        output = self.linear_layer(concatenated)

        return output

@keras.saving.register_keras_serializable(package="transformer_layers")
class TransformerBlock(keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        # TODO:
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
        self_attn_output = self.self_att(inputs, inputs, inputs, attention_mask=attn_mask, training=training)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.layernorm1(inputs + self_attn_output)

        # 3) computed UNMASKED attention using context
        # 4) residual connection and layer normalization
        attn_output = self.ed_attention(context_sequence, context_sequence, out1)
        attn_output = self.dropout(attn_output)
        out2 = self.norm2(out1 + attn_output)

        # 5) feed forward layer
        # 6) residual layer and layer normalization
        ff_output = self.feed_forward(out2)
        ff_output = self.dropout(ff_output)
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
