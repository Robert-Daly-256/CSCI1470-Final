from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import os


##################################### Load in Models ##############################

base_model_path = "./gemma"
finetuned_path = "./finetuned_model"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  
    device_map="auto"           
)

model = PeftModel.from_pretrained(base_model, finetuned_path)

print("\n\nFine-tuned Model Loaded. Prompting.\n\n")

# Note, if loading the model on a MacBook fails. Just run the exact same command,
# python inference.py, again and it will run fine. No idea why this happens.
# no idea why running it again makes this work. 





#################################### Format Prompt ################################

system_prompt = "You are a creative writing assistant. " \
                 "Given a writing prompt, create a story."

writing_prompt = "Writing Prompt: woman in a helment rock climbing."

formatted_prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": writing_prompt}
    ],
    tokenize=False,
    add_generation_prompt=True
)

input = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)



################################ Run Prompt ########################################

output = None
with torch.no_grad():
    output = model.generate(
        **input, 
        max_new_tokens=256,
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n\n\n", response)