import torch
from collections import Counter
import pickle


class Vocab:
    # The vocabulary used by our small language model, 
    # including special words like denotations for the start and end of
    # sentences as well as indications of unknown tokens and padding
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.counter = Counter()
        self.words = []
        
        self.padding_word = '<PAD>'
        self.unknown_word = '<UNK>'
        self.start_word = '<START>'
        self.end_word = '<STOP>'
        
        self.add_word(self.padding_word)
        self.add_word(self.unknown_word)
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        
        self.padding_index = self.word2id[self.padding_word]
        self.unknown_index = self.word2id[self.unknown_word]
        self.start_index = self.word2id[self.start_word]
        self.end_index = self.word2id[self.end_word]
    
    def add_word(self, word):
        # Adds a word to the vocab
        if word not in self.word2id:
            idx = len(self.words)
            self.word2id[word] = idx
            self.id2word[idx] = word
            self.words.append(word)
        return self.word2id[word]
    
    def update_word_counts(self, filename: str):
        # For each occurance of a word in each (potentially clipped) sentence in a file, 
        # update the count of that word in the vocab
        with open(filename, 'r', encoding='utf-8') as dataset:
            for line in dataset:
                sentence = line.strip().lower().split()
                self.counter.update(sentence)
    
    def create_vocab(self, filepaths: list[str], max_vocab_size=1000000, min_word_frequency=50):
        # Creates the final vocab, based on the list of filepaths
        for file in filepaths:
            self.update_word_counts(file)
        
        # Use the word counts to determine which words to add to vocabulary
        word_counts = self.counter.items()
        words_to_add_to_vocab = [(w, c) for w, c in word_counts if c >= min_word_frequency]
        words_to_add_to_vocab = sorted(words_to_add_to_vocab, key= lambda x: x[1], reverse=True)
        words_to_add_to_vocab = [w for w, c in words_to_add_to_vocab[0:max_vocab_size]]
        
        # Add the words to the vocab
        for word in words_to_add_to_vocab:
            self.add_word(word)
    
    def word_to_id(self, word: str) -> int:
        # Takes a word to it's corresponding id in the given vocab
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id[self.unknown_word]
    
    def id_to_word(self, id) -> str:
        # Takes an id to it's corresponding word
        return self.id2word[id]
    
    def encode_sentence(self, sentence: list[str], max_story_length = 1024) -> list[int]:
        # Encodes a story (with padding) clipping to max_length
        encoded_sentence = [self.start_index]
        for word in sentence[:max_story_length - 2]:
            encoded_sentence.append(self.word_to_id(word))
        encoded_sentence.append(self.end_index)
        assert(len(encoded_sentence) <= max_story_length)
        while (len(encoded_sentence) < max_story_length):
            encoded_sentence.append(self.padding_index) 
        return encoded_sentence
    
    def decode_sentence(self, encoded_story: torch.Tensor) -> list[str]:
        decoded_sentence = []
        for x in encoded_story:
            x = int(x.item())
            if x != self.padding_index:
                decoded_sentence.append(self.id_to_word(x))
        return decoded_sentence
            
    
    def preprocess_dataset(self, prompts_filepath: str, prompts_output_filepath: str, stories_filepath: str, stories_output_filepath: str, max_story_length = 1024):
        # Saves a preprocessed dataset to desired location, formated as expected by torch
        preprocessed_prompts = []
        preprocessed_stories = []
        with open(prompts_filepath, 'r', encoding='utf-8') as prompts, open(stories_filepath, 'r', encoding='utf-8') as stories:
            for prompt, story in zip(prompts, stories):
                prompt = prompt.strip().lower().split()
                story = story.strip().lower().split()
                if len(prompt) <= max_story_length - 2 and len(story) <= max_story_length - 2:
                    encoded_prompt = self.encode_sentence(prompt, max_story_length=max_story_length)
                    encoded_story = self.encode_sentence(story, max_story_length=max_story_length)
                    preprocessed_prompts.append(torch.tensor(encoded_prompt, dtype=torch.long))
                    preprocessed_stories.append(torch.tensor(encoded_story, dtype=torch.long))
        
        prompts_dataset = torch.stack(preprocessed_prompts)
        stories_dataset = torch.stack(preprocessed_stories)
        torch.save(prompts_dataset, prompts_output_filepath)
        torch.save(stories_dataset, stories_output_filepath)
        return (prompts_dataset, stories_dataset)
    
    def __len__(self):
        return len(self.words)



if __name__ == "__main__":
    # Data Locations | May need to be editted
    X_train = "./data/original/train.wp_source"
    Y_train = "./data/original/train.wp_target"
    X_test = "./data/original/test.wp_source"
    Y_test = "./data/original/test.wp_target"
    X_val = "./data/original/valid.wp_source"
    Y_val = "./data/original/valid.wp_target"

    # Data Save Locations | May need to be editted
    X_train_save = "./data/preprocessed/X_train.pt"
    Y_train_save = "./data/preprocessed/Y_train.pt"
    X_test_save = "./data/preprocessed/X_test.pt"
    Y_test_save = "./data/preprocessed/Y_test.pt"
    X_val_save = "./data/preprocessed/X_val.pt"
    Y_val_save = "./data/preprocessed/Y_val.pt"

    # Create Vocabs
    vocab = Vocab()
    vocab.create_vocab([X_train, X_val, Y_train, Y_val])
    with open("../vocab.pkl", "wb") as vocab_save:
        pickle.dump(vocab, vocab_save)

    # Preprocess all files
    X_train, Y_train = vocab.preprocess_dataset(X_train, X_train_save, Y_train, Y_train_save)
    X_test, Y_test = vocab.preprocess_dataset(X_test, X_test_save, Y_test, Y_test_save)
    X_val, Y_val = vocab.preprocess_dataset(X_val, X_val_save, Y_val, Y_val_save)

    print(vocab.decode_sentence(X_train[0]))
    print(vocab.decode_sentence(Y_train[0]))
    print("\n\n")
    print(len(vocab))
    print(X_train.size())