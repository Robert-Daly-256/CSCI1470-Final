from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "google/gemma-3-1b-it"
save_path = "./gemma"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)