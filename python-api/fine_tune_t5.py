import os
import time
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
)
from datasets import load_dataset
import torch
from huggingface_hub import HfFolder

# Save your token in an environment variable or use it directly
hf_token = "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"
HfFolder.save_token(hf_token)

# Device setup to always use CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Text Generation Dataset
t5_datasets = [
    ('Salesforce/wikitext', 'wikitext-103-raw-v1')
]

# Function to load dataset with retries
def load_dataset_with_retries(name, config, retries=5, delay=10):
    for attempt in range(retries):
        try:
            return load_dataset(name, config, token=hf_token)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Could not load dataset.")
                raise

# Load datasets
def load_datasets(dataset_names):
    datasets = []
    for name, config in dataset_names:
        try:
            print(f"Loading dataset: {name}")
            datasets.append((name, load_dataset_with_retries(name, config)))
            print(f"Loaded dataset: {name} successfully")
        except Exception as e:
            print(f"Failed to load dataset: {name} with error: {e}")
    return datasets

# Fine-tune T5 for text generation
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

def preprocess_t5_function(examples):
    if 'input_text' in examples and 'target_text' in examples:
        inputs = examples['input_text']
        targets = examples['target_text']
        model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
        with t5_tokenizer.as_target_tokenizer():
            labels = t5_tokenizer(targets, max_length=512, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    else:
        print("Examples:", examples)  # Debug print to see what the examples contain
        raise ValueError("Expected fields 'input_text' and 'target_text' not found in examples")

def fine_tune_t5(datasets):
    for name, dataset in datasets:
        try:
            print(f"Processing dataset: {name}")
            tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
            t5_training_args = TrainingArguments(
                output_dir='./t5_results',
                evaluation_strategy="epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=3,
                weight_decay=0.01,
                use_mps_device=False  # Ensure this parameter is set to False
            )
            t5_trainer = Trainer(
                model=t5_model,
                args=t5_training_args,
                train_dataset=tokenized_t5_dataset['train'],
                eval_dataset=tokenized_t5_dataset['validation']
            )
            t5_trainer.train()
            t5_model.save_pretrained(f'./fine_tuned_t5_{name}')
            t5_tokenizer.save_pretrained(f'./fine_tuned_t5_{name}')
            print(f"Fine-tuned T5 on dataset: {name}")
        except Exception as e:
            print(f"Failed to fine-tune T5 on dataset: {name} with error: {e}")

# Handle potential download timeouts by splitting the dataset loading and training
def main():
    try:
        # Load datasets with retries
        print("Loading text generation datasets...")
        t5_datasets_loaded = load_datasets(t5_datasets)
        
        # Fine-tune models
        print("Fine-tuning T5 for text generation...")
        fine_tune_t5(t5_datasets_loaded)
        
        print("All models have been fine-tuned successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
