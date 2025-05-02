import torch
import transformers
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


#################################### CUDA Clear Memory ##############################################
torch.cuda.empty_cache()



##################################### Model Loading  ################################################
save_path = "./gemma"

lora_config = LoraConfig(
    r=8,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(save_path, attn_implementation='eager', quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(save_path)






################################ DATASET LOADING ####################################################
dataset = datasets.load_from_disk("./data/preprocessed/train_dataset")
dataset = dataset.train_test_split(test_size=0.5)["train"] # Dropped due to GPU memory constraints, which we really shouldn't have had to do.

system_message = "You are a creative writing assistant. " \
                 "Given a writing prompt, create a story."

def format(datapoint):
    prompt = "Writing Prompt: " + str(datapoint["prompt"])
    story = datapoint["story"]
    return {
        "messages": [
            {"role": "system",    "content": system_message},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": story},
        ]
    }

dataset = dataset.map(format, remove_columns=dataset.column_names, batched=False)
dataset = dataset.train_test_split(test_size=0.1)

# Check that data is included as expected
print(dataset["train"][0]["messages"])


#################################### CUDA Clear Memory ##############################################
# Because even an H100 struggles to handle this. LoRA is disapointingly optimized for large datasets
torch.cuda.empty_cache() 




########################################### Fine Tuning #############################################
sft_config = SFTConfig(
    # Training Hyperparameters
    per_device_train_batch_size=8,
    num_train_epochs=5,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=1e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    max_seq_length=2048,

    # Logging
    logging_dir="./logs",
    logging_strategy="steps",
    logging_first_step=True,
    logging_steps=1000,
    log_level="info",

    # Checkpointing
    output_dir="./true_checkpoints",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    args=sft_config,
    peft_config=lora_config,
)


trainer.train()

trainer.model.save_pretrained("./finetuned_model")