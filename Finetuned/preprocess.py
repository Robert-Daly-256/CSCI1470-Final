import torch
from collections import Counter
import pickle
import re
import datasets
from datasets import Dataset

    
def clean(text: str) -> str:
    # Take tokenized data to untokenized data, to be retokenized by Gemma. Is this inefficient, yes, but it works.
    text = re.sub(r'``\s', '``', text)                        # | Transormation:  `` hi -> "hi
    text = re.sub(r'\s\'\'', "''", text)                      # | Transormation: hi '' -> hi''
    text = re.sub(r'<newline>', '\n', text)                   # | Transormation: <newline> -> \n
    text = re.sub(r'``|\'\'', '"', text)                      # | Transormation: '' -> "
    text = re.sub(r"\b(\w+)\s'(\w+)\b", r"\1'\2", text)       # | Transormation: <text1> '<text2> -> <text1>'<text2>, i.e., we 're -> we're
    text = re.sub(r"\b(\w+)\s+n[']?t\b", r"\1n't", text)      # | Transormation: was n't -> wasn't
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)              # | Transormation: hi . -> hi.
    text = re.sub(r'([.,!?])(?=\w)', r'\1 ', text)            # | Transormation: hi.bob -> hi. bob
    return text
        

def preprocess_dataset(prompts_filepath: str, stories_filepath: str, max_story_length = 1000):
    # Saves a preprocessed dataset to desired location, formated as expected by Hugging Face Ecosystem.
    preprocessed = []
    with open(prompts_filepath, 'r', encoding='utf-8') as prompts, open(stories_filepath, 'r', encoding='utf-8') as stories:
        for prompt, story in zip(prompts, stories):
            prompt_array = prompt.strip().split()
            story_array = story.strip().split()
            if len(prompt_array) <= max_story_length - 2 and len(story_array) <= max_story_length - 2:
                string_prompt = clean(prompt)
                string_story = clean(story)
                preprocessed.append({"prompt": string_prompt, "story": string_story})

    return Dataset.from_list(preprocessed)




if __name__ == "__main__":
    # Data Locations | May need to be editted
    X_train = "./data/original/train.wp_source"
    Y_train = "./data/original/train.wp_target"
    X_test = "./data/original/test.wp_source"
    Y_test = "./data/original/test.wp_target"
    X_val = "./data/original/valid.wp_source"
    Y_val = "./data/original/valid.wp_target"

    # Data Save Locations | May need to be editted
    train_save = "./data/preprocessed/train.pkl"
    test_save = "./data/preprocessed/test.pkl"
    val_save = "./data/preprocessed/val.pkl"

    # Preprocess all files
    train = preprocess_dataset(X_train, Y_train)
    test = preprocess_dataset(X_test, Y_test)
    val = preprocess_dataset(X_val, Y_val)

    # Save to disk
    train.save_to_disk("./data/preprocessed/train_dataset")
    test.save_to_disk("./data/preprocessed/test_dataset")
    val.save_to_disk("./data/preprocessed/val_dataset")
