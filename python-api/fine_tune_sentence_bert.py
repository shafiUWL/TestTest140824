import os
import time
from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from huggingface_hub import HfFolder

# Save your token in an environment variable or use it directly
hf_token = "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"
HfFolder.save_token(hf_token)

# Device setup to always use CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Sentence Similarity Dataset
sentence_bert_datasets = [
    ('sentence-transformers/stsb', None)
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

# Fine-tune Sentence-BERT for sentence similarity
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

def preprocess_sentence_bert_function(dataset):
    if 'sentence1' in dataset['train'][0] and 'sentence2' in dataset['train'][0]:
        sentence_examples = [
            InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label'])
            for row in dataset['train']
        ]
        return sentence_examples
    else:
        print("Examples:", dataset['train'][0])  # Debug print to see what the examples contain
        raise ValueError("Expected fields 'sentence1' and 'sentence2' not found in examples")

def fine_tune_sentence_bert(datasets):
    for name, dataset in datasets:
        try:
            print(f"Processing dataset: {name}")
            sentence_examples = preprocess_sentence_bert_function(dataset)
            sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
            sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

            sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
            sentence_model.save(f'fine_tuned_sentence_bert_{name}')
            print(f"Fine-tuned Sentence-BERT on dataset: {name}")
        except Exception as e:
            print(f"Failed to fine-tune Sentence-BERT on dataset: {name} with error: {e}")

# Handle potential download timeouts by splitting the dataset loading and training
def main():
    try:
        # Load datasets with retries
        print("Loading sentence similarity datasets...")
        sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
        # Fine-tune models
        print("Fine-tuning Sentence-BERT for sentence similarity...")
        fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
        print("All models have been fine-tuned successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
