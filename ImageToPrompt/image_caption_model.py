import numpy as np
import tensorflow as tf
import keras
@keras.saving.register_keras_serializable(package="ImCapModel")
class ImageCaptionModel(keras.Model):

    def __init__(self, decoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder

    
    def call(self, encoded_images, captions):
        return self.decoder(encoded_images, captions)  

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]

    def train(self, train_captions, train_image_features, padding_index, batch_size=30):
        """
        Runs through one epoch - all training examples.

        :param model: the initialized model to use for forward and backward pass
        :param train_captions: train data captions (all data for training) 
        :param train_image_features: train image features (all data for training) 
        :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
        :return: None
        """
        # NOTE: 
        # - The captions passed to the decoder should have the last token in the window removed:
        #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
        #
        # - When computing loss, the decoder labels should have the first word removed:
        #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]
        #    Additionally, you should create a boolean mask to ignore the padding labels in the loss function.
        #	    The mask should be 0 for padding tokens in the label and 1 for all other tokens.
        #		[<START> student working on homework *PAD*] --> [1 1 1 1 1 0]
        ## HINT: shuffle the training examples (perhaps using tf.random.shuffle on a
        ##      range of indices spanning # of training entries, then tf.gather) 
        ##      to make training smoother over multiple epochs.

        n = len(train_captions)
        indices = tf.random.shuffle(tf.range(n))
        my_captions = tf.gather(train_captions, indices)
        my_image_feats = tf.gather(train_image_features, indices)
        
        for i, end in enumerate(range(batch_size, len(my_captions)+1, batch_size)):
            print(i, end="\r")

            start = end - batch_size
            batch_image_features = my_image_feats[start:end, :]
            decoder_input = my_captions[start:end, :-1]
            decoder_labels = my_captions[start:end, 1:]

            mask = decoder_labels != padding_index
            mask_float = tf.cast(mask, tf.float32)

            with tf.GradientTape() as tape:
                predictions = self(batch_image_features, decoder_input, training=True)
                loss = self.loss_function(predictions, decoder_labels, mask_float)

            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))




    def test(self, test_captions, test_image_features, padding_index, batch_size=30):
        """
        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_captions: test caption data (all data for testing) of shape (num captions,20)
        :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
        :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
        :returns: perplexity of the test set, per symbol accuracy on test set
        """
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            batch_image_features = test_image_features[start:end, :]
            decoder_input = test_captions[start:end, :-1]
            decoder_labels = test_captions[start:end, 1:]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            probs = self(batch_image_features, decoder_input)
            mask = decoder_labels != padding_index
            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            loss = self.loss_function(probs, decoder_labels, mask)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()        
        return avg_prp, avg_acc

    def get_config(self):
        base_config = super().get_config()
        config = {
            "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        decoder_config = config.pop("decoder")
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
        return cls(decoder, **config)


def accuracy_function(prbs, labels, mask):
    """
    DO NOT CHANGE

    Computes the batch accuracy

    :param prbs:  float tensor, word prediction probabilities [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
    :param labels:  integer tensor, word prediction labels [BATCH_SIZE x WINDOW_SIZE]
    :param mask:  tensor that acts as a padding mask [BATCH_SIZE x WINDOW_SIZE]
    :return: scalar tensor of accuracy of the batch between 0 and 1
    """

    correct_classes = tf.cast(tf.argmax(prbs, axis=-1), tf.int32) == tf.cast(labels, tf.int32)
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
    return accuracy


def loss_function(prbs, labels, mask):
    """
    Calculates the model cross-entropy loss after one forward pass
    Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

    :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
    :param labels:  integer tensor, word prediction labels [batch_size x window_size]
    :param mask:  tensor that acts as a padding mask [batch_size x window_size]
    :return: the loss of the model as a tensor
    """
    masked_labs = tf.boolean_mask(labels, mask)
    masked_prbs = tf.boolean_mask(prbs, mask)
    scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
    loss = tf.reduce_sum(scce)
    return loss

# import torch
# import torch.nn as nn

# class PositionalEncoding(nn.Module):
#     def __init__(self, model_dim, max_len=5000):
#         super().__init__()
#         positional_encoding = torch.zeros(max_len, model_dim)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, model_dim, 2) * (-torch.log(torch.tensor(10000.0)) / model_dim)
#         )
#         positional_encoding[:, 0::2] = torch.sin(position * div_term)
#         positional_encoding[:, 1::2] = torch.cos(position * div_term)
#         self.pe = positional_encoding.unsqueeze(0)  # (1, max_len, d_model)

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)].to(x.device)

# # encoder class --> ask Dave if that's okay 
# # using CNN since good at extracting spatial features from images --> might also consider using a transformer but idk 
# class CNNEncoder(nn.Module):
#     def __init__(self, model_dim):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#         )
#         self.linear = nn.Linear(128, model_dim)
#         self.batch_norm = nn.BatchNorm1d(model_dim, momentum=0.01)

#     def forward(self, images):
#         features = self.cnn(images).view(images.size(0), -1)
#         features = self.linear(features)
#         return self.batch_norm(features)

# # transformer decoder
# # chose transformer because good at sequential info 
# class TransformerDecoder(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, pad_idx):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
#         self.pos_encoding = PositionalEncoding(d_model)
#         decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
#         self.fc_out = nn.Linear(d_model, vocab_size)

#     def forward(self, target, memory, target_mask=None, target_key_padding_mask=None):
#         target_embedding = self.embedding(target) * (memory.size(-1) ** 0.5)
#         target_embedding = self.pos_encoding(target_embedding)
#         target_embedding = target_embedding.transpose(0, 1)  # (T, N, E)
#         memory = memory.unsqueeze(0)  # (1, N, E)
#         output = self.transformer_decoder(
#             target_embedding, memory, tgt_mask=target_mask, tgt_key_padding_mask=target_key_padding_mask
#         )
#         return self.fc_out(output.transpose(0, 1))  # (N, T, vocab_size)


# class ImageCaptioningModel(nn.Module):
#     def __init__(self, vocab_size, model_dim=512, nhead=8, num_layers=6, dim_feedforward=2048, pad_idx=0):
#         super().__init__()
#         self.encoder = CNNEncoder(model_dim)
#         self.decoder = TransformerDecoder(vocab_size, model_dim, nhead, num_layers, dim_feedforward, pad_idx)

#     def forward(self, images, captions, tgt_mask=None, tgt_key_padding_mask=None):
#         memory = self.encoder(images)  # (batch_size, model_dim)
#         output = self.decoder(captions, memory, tgt_mask, tgt_key_padding_mask)
#         return output
