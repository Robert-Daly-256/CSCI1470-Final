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

    # def get_build_config(self):
    #     # base_config = super().get_config()
    #     config = {
    #         "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
    #     }
    #     return config

    @classmethod
    def from_config(cls, config):
        decoder_config = config.pop("decoder")
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
        # print("i did it")
        return cls(decoder, **config)
    
    # def build_from_config(self, config):
    #     decoder_config = config.pop("decoder")
    #     self.decoder = tf.keras.utils.deserialize_keras_object(decoder_config)

    #     if not self.built:
    #         self.build(input_shape=(None, 2048))
    
    def predict_caption(self, image_feature, word2idx, max_length=20, start_token='<start>', stop_token='<end>'):
        """
            image_feature: Tensor of shape (1, 2048) or similar
            vocab: vocab object from .pkl file
            max_length: maximum caption length
            start_token: special start token
            stop_token: special stop token

        outputs the generated caption
        """
        # convert start and stop tokens to their IDs in the vocab
        start_token_id = word2idx.get(start_token, None)
        stop_token_id = word2idx.get(stop_token, None)

        idx2word = {idx: word for word, idx in word2idx.items()}

        if start_token_id is None:
            raise ValueError(f"'{start_token}' token not found in vocab")
        if stop_token_id is None:
            raise ValueError(f"'{stop_token}' token not found in vocab")
        
        caption_tokens = [start_token_id]

        # dummy print to check vocab
        # dummy_token_ids = list(idx2word.keys())[:5]  # take 5 token IDs
        # dummy_caption_words = [idx2word.get(id, '') for id in dummy_token_ids]
        # dummy_caption = ' '.join(dummy_caption_words)
        # print("Dummy caption:", dummy_caption)

        for _ in range(max_length):
            input_caption = tf.expand_dims(caption_tokens, axis=0)  # (1, current_length)
            
            preds = self.decoder(image_feature, input_caption)  # (1, current_length, vocab_size)
            # print("Decoder output preds:", preds)
            # print("Preds shape:", preds.shape)
            preds = preds[:, -1, :]  # last token's prediction 

            # choose the highest-probability token
            next_token_id = tf.argmax(preds, axis=-1).numpy()[0]  
            
            if next_token_id == stop_token_id:
                break
            
            caption_tokens.append(next_token_id)

        # convert token IDs back to strings/words using the vocab dictionary
        # caption_words = [word2idx.get(id, '') for id in caption_tokens[1:]] 
        caption_words = [idx2word.get(id, '') for id in caption_tokens[1:]]  # skip start token
        caption = ' '.join(caption_words)
        
        return caption


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