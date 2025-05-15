# python3.10 -m pip install --upgrade bitsandbytes transformers peft accelerate datasets trl huggingface_hub
# python3.10 -m pip install auto-round
# jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
# python3.10 -m pip install --upgrade bitsandbytes transformers peft accelerate datasets trl huggingface_hub
# python3.10 -m pip install auto-round
# jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
# https://github.com/huggingface/accelerate/issues/1239

import argparse
import json
import os

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

from logging_class import start_queue, stop_log, write_log

from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')
wandb.login('allow',"69b9681e7dc41d211e8c93a3ba9a6fb8d781404a")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser(description='AIxBlock')
parser.add_argument('--training_args_json', type=str, default=None, help="JSON string for training arguments")
parser.add_argument('--dataset_local', type=str, default=None, help="dataset id")
parser.add_argument('--channel_log', type=str, default=None, help="channel_log")
parser.add_argument('--hf_model_id', type=str, default=None, help="hf_model_id")
parser.add_argument('--push_to_hub', type=str, default=None, help="push_to_hub")
parser.add_argument('--push_to_hub_token', type=str, default=None, help="push_to_hub_token")

# Phân tích các tham số dòng lệnh
args = parser.parse_args()

log_queue, logging_thread = start_queue(args.channel_log)
write_log(log_queue)
dataset_local = args.dataset_local
is_use_local = False
num_train_epochs = 1
per_train_dataset = 0.8
per_test_dataset = 0.2
output_dir="./data/checkpoint"

push_to_hub= True if args.push_to_hub and args.push_to_hub == "True" else False
hf_model_id = args.hf_model_id if args.hf_model_id else "deepseek-r1-4b"
push_to_hub_token = args.push_to_hub_token if args.push_to_hub_token else "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"

print(push_to_hub, hf_model_id, push_to_hub_token)
# Nếu có file JSON, đọc và phân tích nó
print(args.training_args_json)
if args.training_args_json:
    with open(args.training_args_json, 'r') as f:
        training_args_dict = json.load(f)
else:
    training_args_dict = {}

#use bf16 and FlashAttention if supported
if torch.cuda.is_bf16_supported():
#   os.system('python3.10 -m pip install flash_attn')
  compute_dtype = torch.bfloat16
  attn_implementation = 'flash_attention_2'
else:
  compute_dtype = torch.float16
  attn_implementation = 'sdpa'
torch.set_grad_enabled(True)
model_name = "tonyshark/deepseek-r1-4b"
#Tokenizer
print("Download tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id =  tokenizer.eos_token_id
# tokenizer.padding_side = 'left'
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {}

                ### Input:
                {}

                ### Response:
                {}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def tokenizer_func_text(examples):
    texts = examples["text"]  # Trường 'text' chứa dữ liệu đầu vào
    return tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=128, 
        return_tensors="pt"  # Trả về tensor PyTorch, bỏ đi nếu không cần
    )

def tokenizer_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]

    # outputs      = examples["text"]
    # outputs      = examples["input_ids"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return tokenizer("".join(texts),truncation=True, padding=True, max_length=128, return_tensors="pt")

dataset_id = "goutampunasiya/pretraining-data-stories-input-ids"
if not training_args_dict:
    training_args_dict = {}
                # examples follow format of resp json files
 # train_dataset = ds.map(tokenizer_func,remove_columns=ds.column_names, batched=True)
if "dataset_id" in training_args_dict:
    dataset_id= training_args_dict["dataset_id"] #"Sujithanumala/Llama_3.2_1B_IT_dataset"
else:
    dataset_id = "goutampunasiya/pretraining-data-stories-input-ids"

    if dataset_local and dataset_local!="None":

        dataset_id = dataset_local
        is_use_local = True

if "num_train_epochs" in training_args_dict:
    num_train_epochs = int(training_args_dict["num_train_epochs"])

if "per_train_dataset" in training_args_dict:
    per_train_dataset = int(training_args_dict["per_train_dataset"])

if "per_test_dataset" in training_args_dict:
    per_test_dataset = int(training_args_dict["per_test_dataset"])
    
sfttrainer_args={}
def formatting_func(example):
    text = (f"{example['instruction']} {example['input']} {example['output']}")
    return {"text" : text}

if not is_use_local:
    dataset = load_dataset(dataset_id)
    # Truy cập từng tập dữ liệu
    train_test_split = dataset["train"].train_test_split(test_size=per_test_dataset, seed=42)  # 20% cho eval
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # train_dataset = dataset["train"]
    # eval_dataset = dataset["train"]

    if "input_ids" in train_dataset.features.keys():
        print(train_dataset.features)
    elif "text" in  train_dataset.features.keys():
        print(train_dataset.features)
        # sfttrainer_args={
        #     "dataset_text_field":"text"
        # }
        train_dataset = train_dataset.map(tokenizer_func_text,remove_columns=train_dataset.column_names, batched=True)
        eval_dataset = eval_dataset.map(tokenizer_func_text,remove_columns=eval_dataset.column_names, batched=True)
    elif "instruction" in  train_dataset.features.keys() and  "input" in  train_dataset.features.keys() and  "output" in  train_dataset.features.keys():
        print(train_dataset.features)
        sfttrainer_args={
            formatting_func:formatting_func,
            "dataset_text_field":"text"
        }
        train_dataset = train_dataset.map(formatting_func)
        eval_dataset = eval_dataset.map(formatting_func)
        
        train_dataset = train_dataset.map(tokenizer_func,remove_columns=train_dataset.column_names, batched=True)
        eval_dataset = eval_dataset.map(tokenizer_func,remove_columns=eval_dataset.column_names, batched=True)
  
else:
    # Load dataset từ thư mục local
    dataset = load_dataset(
        "json", 
        data_files={
            "train": f"{dataset_id}/train_data.json",
            "test": f"{dataset_id}/test_data.json"
        }
    )

    # Truy cập từng tập dữ liệu
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    if "input_ids" in train_dataset.features.keys():
        print(train_dataset.features)
    elif "text" in  train_dataset.features.keys():
        print(train_dataset.features)
        sfttrainer_args={
            "dataset_text_field":"text"
        }
        train_dataset = train_dataset.map(tokenizer_func_text,remove_columns=train_dataset.column_names, batched=True)
        eval_dataset = eval_dataset.map(tokenizer_func_text,remove_columns=eval_dataset.column_names, batched=True)
        
    elif "instruction" in  train_dataset.features.keys() and  "input" in  train_dataset.features.keys() and  "output" in  train_dataset.features.keys():
        print(train_dataset.features)
        sfttrainer_args={
            formatting_func:formatting_func,
            "dataset_text_field":"text"
        }
        train_dataset = train_dataset.map(formatting_func)
        eval_dataset = eval_dataset.map(formatting_func)
        
        train_dataset = train_dataset.map(tokenizer_func,remove_columns=train_dataset.column_names, batched=True)
        eval_dataset = eval_dataset.map(tokenizer_func,remove_columns=eval_dataset.column_names, batched=True)
    
bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
#, attn_implementation=attn_implementation
# quantization_config=bnb_config,
model = AutoModelForCausalLM.from_pretrained(
          model_name,  device_map={"": 0}, trust_remote_code=True
)

# print(model)
# model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= "all-linear"
)

training_arguments = TrainingArguments(
        output_dir="./data/checkpoint", 
        eval_strategy="steps",
        do_eval=True,
        # optim="paged_adamw_8bit",
        # per_device_train_batch_size=8,
        # gradient_accumulation_steps=4,
        # per_device_eval_batch_size=8,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=1e-4,
        # fp16 = not torch.cuda.is_bf16_supported(),
        # bf16 = torch.cuda.is_bf16_supported(),
        eval_steps=10,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        remove_unused_columns=False,
        report_to="tensorboard", #azure_ml, comet_ml, mlflow, neptune, tensorboard, wandb, codecarbon, clearml, dagshub, flyte, dvclive
        push_to_hub = push_to_hub,
        push_to_hub_model_id=hf_model_id,#[project-id]-[model-name]-[datetime]
        push_to_hub_token=push_to_hub_token,
)
# https://stackoverflow.com/questions/78688141/how-to-choose-dataset-text-field-in-sfttrainer-hugging-face-for-my-llm-model

trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # peft_config=peft_config,
        # **sfttrainer_args,
        # max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments
        # dataset_kwargs={
        #                 "add_special_tokens": False,  # We template with special tokens
        #                 "append_concat_token": False, # No need to add additional separator token
        #                 'skip_prepare_dataset': True # skip the dataset preparation
        #             },
)

trainer.train()

if push_to_hub:
    trainer.push_to_hub()
# save model
# MODEL_DIR = os.getenv('MODEL_DIR', './data/checkpoint')
# FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME',hf_model_id)
# chk_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
# print(f"Model is trained and saved as {chk_path}")
# trainer.save_model(chk_path)
# push to hub

# free the memory again
del model
del trainer
torch.cuda.empty_cache()