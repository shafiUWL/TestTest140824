import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
import torch

# Load BERT dataset
bert_df = pd.read_csv('bert_data.csv')  # Ensure this CSV file has 'text' and 'label' columns
bert_dataset = DatasetDict({
    'train': Dataset.from_pandas(bert_df.sample(frac=0.8, random_state=42)),  # 80% of data for training
    'eval': Dataset.from_pandas(bert_df.drop(bert_df.sample(frac=0.8, random_state=42).index))  # 20% of data for evaluation
})

# Tokenizer and model for BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess function for BERT dataset
def bert_preprocess_function(examples):
    return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# Tokenize the BERT dataset
tokenized_bert_dataset = bert_dataset.map(bert_preprocess_function, batched=True)
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Load BERT model

# Training arguments for BERT
bert_training_args = TrainingArguments(
    output_dir='./bert_results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    use_cpu=True,
)

# Trainer for BERT
bert_trainer = Trainer(
    model=bert_model,
    args=bert_training_args,
    train_dataset=tokenized_bert_dataset['train'],
    eval_dataset=tokenized_bert_dataset['eval']
)

# Fine-tune BERT model
bert_trainer.train()
# Save the fine-tuned BERT model
bert_model.save_pretrained('./fine_tuned_bert')

# Load Sentence-BERT dataset
sentence_bert_df = pd.read_csv('sentence_similarity.csv')  # Ensure this CSV file has 'sentence1', 'sentence2', 'label' columns
sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for _, row in sentence_bert_df.iterrows()]

# Define Sentence-BERT model and dataloader
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

# Fine-tune Sentence-BERT model
sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
# Save the fine-tuned Sentence-BERT model
sentence_model.save('fine_tuned_sentence_bert')

# Load T5 dataset
t5_dataset = load_dataset('t5_data.csv')  # Ensure this dataset has 'input_text' and 'target_text' columns

# Tokenizer and model for T5
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Preprocess function for T5 dataset
def t5_preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
    with t5_tokenizer.as_target_tokenizer():
        labels = t5_tokenizer(targets, max_length=512, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize the T5 dataset
tokenized_t5_dataset = t5_dataset.map(t5_preprocess_function, batched=True)

# Training arguments for T5
t5_training_args = TrainingArguments(
    output_dir='./t5_results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer for T5
t5_trainer = Trainer(
    model=t5_model,
    args=t5_training_args,
    train_dataset=tokenized_t5_dataset['train'],
    eval_dataset=tokenized_t5_dataset['eval']
)

# Fine-tune T5 model
t5_trainer.train()
# Save the fine-tuned T5 model
t5_model.save_pretrained('fine_tuned_t5')
# Save the tokenizer for the fine-tuned T5 model
t5_tokenizer.save_pretrained('fine_tuned_t5')










































# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch

# # Load BERT dataset
# bert_df = pd.read_csv('bert_data.csv')  # Ensure this CSV file has 'text' and 'label' columns
# bert_dataset = DatasetDict({
#     'train': Dataset.from_pandas(bert_df.sample(frac=0.8, random_state=42)),
#     'eval': Dataset.from_pandas(bert_df.drop(bert_df.sample(frac=0.8, random_state=42).index))
# })

# # Tokenizer and model for BERT
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def bert_preprocess_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# tokenized_bert_dataset = bert_dataset.map(bert_preprocess_function, batched=True)
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Training arguments for BERT
# bert_training_args = TrainingArguments(
#     output_dir='./bert_results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     use_cpu=True,
# )

# # Trainer for BERT
# bert_trainer = Trainer(
#     model=bert_model,
#     args=bert_training_args,
#     train_dataset=tokenized_bert_dataset['train'],
#     eval_dataset=tokenized_bert_dataset['eval']
# )

# # Fine-tune BERT
# bert_trainer.train()
# bert_model.save_pretrained('./fine_tuned_bert')

# # Load Sentence-BERT dataset
# sentence_bert_df = pd.read_csv('sentence_similarity.csv')  # Ensure this CSV file has 'sentence1', 'sentence2', 'label' columns
# sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for _, row in sentence_bert_df.iterrows()]

# # Define Sentence-BERT model and dataloader
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
# sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

# # Fine-tune Sentence-BERT
# sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
# sentence_model.save('fine_tuned_sentence_bert')

# # Load T5 dataset
# t5_dataset = load_dataset('t5_data.csv')  # Ensure this dataset has 'input_text' and 'target_text' columns

# # Tokenizer and model for T5
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# def t5_preprocess_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# tokenized_t5_dataset = t5_dataset.map(t5_preprocess_function, batched=True)

# # Training arguments for T5
# t5_training_args = TrainingArguments(
#     output_dir='./t5_results',
#     evaluation_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# # Trainer for T5
# t5_trainer = Trainer(
#     model=t5_model,
#     args=t5_training_args,
#     train_dataset=tokenized_t5_dataset['train'],
#     eval_dataset=tokenized_t5_dataset['eval']
# )

# # Fine-tune T5
# t5_trainer.train()
# t5_model.save_pretrained('fine_tuned_t5')
# t5_tokenizer.save_pretrained('fine_tuned_t5')


# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset, DatasetDict
# import torch

# # Load dataset
# df = pd.read_csv('math.csv')  # Change this to load the correct CSV file for each subject
# df['text'] = df['question'] + ' ' + df['answer'] + ' ' + df['working_out']
# df = df[['text', 'label']]  # Ensure the DataFrame only has the required columns

# # Split the dataset into train and eval
# train_df = df.sample(frac=0.8, random_state=42)  # 80% for training
# eval_df = df.drop(train_df.index)  # 20% for evaluation

# # Convert to Hugging Face Dataset
# dataset = DatasetDict({
#     'train': Dataset.from_pandas(train_df),
#     'eval': Dataset.from_pandas(eval_df)
# })

# # Tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def preprocess_function(examples):
#     if isinstance(examples['text'], list):
#         examples['text'] = [str(x) for x in examples['text']]
#     return tokenizer(examples['text'], padding='max_length', truncation=True)

# tokenized_datasets = dataset.map(preprocess_function, batched=True)

# # Load model
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Force the model to run on CPU
# device = torch.device("cpu")
# model.to(device)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     use_cpu=True,  # Use CPU
# )

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['eval']
# )

# # Train model
# trainer.train()

# # Save the model
# trainer.save_model('./fine_tuned_model')

# import os
# import time
# import pandas as pd
# from transformers import (
#     BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
#     T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, 
#     DataCollatorWithPadding
# )
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Dataset
# bert_datasets = [
#     ('stanfordnlp/imdb', None)
# ]

# # Sentence Similarity Dataset
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None)
# ]

# # Question Answering Dataset
# qa_datasets = [
#     ('squad', None)
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         try:
#             print(f"Loading dataset: {name}")
#             datasets.append((name, load_dataset_with_retries(name, config)))
#             print(f"Loaded dataset: {name} successfully")
#         except Exception as e:
#             print(f"Failed to load dataset: {name} with error: {e}")
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     if 'text' in examples:
#         return bert_tokenizer(examples['text'], padding='max_length', truncation=True)
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected field 'text' not found in examples")

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#             bert_training_args = TrainingArguments(
#                 output_dir='./bert_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=2e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             bert_trainer = Trainer(
#                 model=bert_model,
#                 args=bert_training_args,
#                 train_dataset=tokenized_bert_dataset['train'],
#                 eval_dataset=tokenized_bert_dataset['test']
#             )
#             bert_trainer.train()
#             bert_model.save_pretrained(f'./fine_tuned_bert_{name}')
#             print(f"Fine-tuned BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {name} with error: {e}")

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def preprocess_sentence_bert_function(dataset):
#     if 'sentence1' in dataset['train'][0] and 'sentence2' in dataset['train'][0]:
#         sentence_examples = [
#             InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label'])
#             for row in dataset['train']
#         ]
#         return sentence_examples
#     else:
#         print("Examples:", dataset['train'][0])  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'sentence1' and 'sentence2' not found in examples")

# def fine_tune_sentence_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             sentence_examples = preprocess_sentence_bert_function(dataset)
#             sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#             sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

#             sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#             sentence_model.save(f'fine_tuned_sentence_bert_{name}')
#             print(f"Fine-tuned Sentence-BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune Sentence-BERT on dataset: {name} with error: {e}")

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     if 'question' in examples and 'context' in examples:
#         return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'question' and 'context' not found in examples")

# def fine_tune_qa(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#             qa_training_args = TrainingArguments(
#                 output_dir='./qa_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=3e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             qa_trainer = Trainer(
#                 model=qa_model,
#                 args=qa_training_args,
#                 train_dataset=tokenized_qa_dataset['train'],
#                 eval_dataset=tokenized_qa_dataset['validation'],
#                 data_collator=data_collator,
#             )
#             qa_trainer.train()
#             qa_model.save_pretrained(f'./fine_tuned_qa_{name}')
#             qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{name}')
#             print(f"Fine-tuned BERT for QA on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT for QA on dataset: {name} with error: {e}")

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets(bert_datasets)
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# import time
# import pandas as pd
# from transformers import (
#     BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
#     T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, 
#     DataCollatorWithPadding
# )
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Dataset
# bert_datasets = [
#     ('stanfordnlp/imdb', None)
# ]

# # Sentence Similarity Dataset
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None)
# ]

# # Text Generation Dataset
# t5_datasets = [
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Dataset
# qa_datasets = [
#     ('squad', None)
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         try:
#             print(f"Loading dataset: {name}")
#             datasets.append((name, load_dataset_with_retries(name, config)))
#             print(f"Loaded dataset: {name} successfully")
#         except Exception as e:
#             print(f"Failed to load dataset: {name} with error: {e}")
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     if 'text' in examples:
#         return bert_tokenizer(examples['text'], padding='max_length', truncation=True)
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected field 'text' not found in examples")

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#             bert_training_args = TrainingArguments(
#                 output_dir='./bert_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=2e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             bert_trainer = Trainer(
#                 model=bert_model,
#                 args=bert_training_args,
#                 train_dataset=tokenized_bert_dataset['train'],
#                 eval_dataset=tokenized_bert_dataset['test']
#             )
#             bert_trainer.train()
#             bert_model.save_pretrained(f'./fine_tuned_bert_{name}')
#             print(f"Fine-tuned BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {name} with error: {e}")

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def preprocess_sentence_bert_function(dataset):
#     if 'sentence1' in dataset['train'][0] and 'sentence2' in dataset['train'][0]:
#         sentence_examples = [
#             InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label'])
#             for row in dataset['train']
#         ]
#         return sentence_examples
#     else:
#         print("Examples:", dataset['train'][0])  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'sentence1' and 'sentence2' not found in examples")

# def fine_tune_sentence_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             sentence_examples = preprocess_sentence_bert_function(dataset)
#             sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#             sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

#             sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#             sentence_model.save(f'fine_tuned_sentence_bert_{name}')
#             print(f"Fine-tuned Sentence-BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune Sentence-BERT on dataset: {name} with error: {e}")

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     if 'input_text' in examples and 'target_text' in examples:
#         inputs = examples['input_text']
#         targets = examples['target_text']
#         model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#         with t5_tokenizer.as_target_tokenizer():
#             labels = t5_tokenizer(targets, max_length=512, truncation=True)
#         model_inputs['labels'] = labels['input_ids']
#         return model_inputs
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'input_text' and 'target_text' not found in examples")

# def fine_tune_t5(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#             t5_training_args = TrainingArguments(
#                 output_dir='./t5_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=5e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             t5_trainer = Trainer(
#                 model=t5_model,
#                 args=t5_training_args,
#                 train_dataset=tokenized_t5_dataset['train'],
#                 eval_dataset=tokenized_t5_dataset['validation']
#             )
#             t5_trainer.train()
#             t5_model.save_pretrained(f'./fine_tuned_t5_{name}')
#             t5_tokenizer.save_pretrained(f'fine_tuned_t5_{name}')
#             print(f"Fine-tuned T5 on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune T5 on dataset: {name} with error: {e}")

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     if 'question' in examples and 'context' in examples:
#         return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'question' and 'context' not found in examples")

# def fine_tune_qa(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#             qa_training_args = TrainingArguments(
#                 output_dir='./qa_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=3e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             qa_trainer = Trainer(
#                 model=qa_model,
#                 args=qa_training_args,
#                 train_dataset=tokenized_qa_dataset['train'],
#                 eval_dataset=tokenized_qa_dataset['validation'],
#                 data_collator=data_collator,
#             )
#             qa_trainer.train()
#             qa_model.save_pretrained(f'./fine_tuned_qa_{name}')
#             qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{name}')
#             print(f"Fine-tuned BERT for QA on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT for QA on dataset: {name} with error: {e}")

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets(bert_datasets)
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# import time
# import pandas as pd
# from transformers import (
#     BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
#     T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, 
#     DataCollatorWithPadding
# )
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Dataset
# bert_datasets = [
#     ('stanfordnlp/imdb', None)
# ]

# # Sentence Similarity Dataset
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None)
# ]

# # Text Generation Dataset
# t5_datasets = [
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Dataset
# qa_datasets = [
#     ('squad', None)
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         try:
#             print(f"Loading dataset: {name}")
#             datasets.append((name, load_dataset_with_retries(name, config)))
#             print(f"Loaded dataset: {name} successfully")
#         except Exception as e:
#             print(f"Failed to load dataset: {name} with error: {e}")
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     if 'text' in examples:
#         return bert_tokenizer(examples['text'], padding='max_length', truncation=True)
#     else:
#         raise ValueError("Expected field 'text' not found in examples")

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#             bert_training_args = TrainingArguments(
#                 output_dir='./bert_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=2e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             bert_trainer = Trainer(
#                 model=bert_model,
#                 args=bert_training_args,
#                 train_dataset=tokenized_bert_dataset['train'],
#                 eval_dataset=tokenized_bert_dataset['test']
#             )
#             bert_trainer.train()
#             bert_model.save_pretrained(f'./fine_tuned_bert_{name}')
#             print(f"Fine-tuned BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {name} with error: {e}")

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def preprocess_sentence_bert_function(dataset):
#     if 'sentence1' in dataset['train'][0] and 'sentence2' in dataset['train'][0]:
#         sentence_examples = [
#             InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label'])
#             for row in dataset['train']
#         ]
#         return sentence_examples
#     else:
#         print("Examples:", dataset['train'][0])  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'sentence1' and 'sentence2' not found in examples")

# def fine_tune_sentence_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             sentence_examples = preprocess_sentence_bert_function(dataset)
#             sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#             sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

#             sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#             sentence_model.save(f'fine_tuned_sentence_bert_{name}')
#             print(f"Fine-tuned Sentence-BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune Sentence-BERT on dataset: {name} with error: {e}")

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     if 'input_text' in examples and 'target_text' in examples:
#         inputs = examples['input_text']
#         targets = examples['target_text']
#         model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#         with t5_tokenizer.as_target_tokenizer():
#             labels = t5_tokenizer(targets, max_length=512, truncation=True)
#         model_inputs['labels'] = labels['input_ids']
#         return model_inputs
#     else:
#         raise ValueError("Expected fields 'input_text' and 'target_text' not found in examples")

# def fine_tune_t5(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#             t5_training_args = TrainingArguments(
#                 output_dir='./t5_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=5e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             t5_trainer = Trainer(
#                 model=t5_model,
#                 args=t5_training_args,
#                 train_dataset=tokenized_t5_dataset['train'],
#                 eval_dataset=tokenized_t5_dataset['validation']
#             )
#             t5_trainer.train()
#             t5_model.save_pretrained(f'./fine_tuned_t5_{name}')
#             t5_tokenizer.save_pretrained(f'fine_tuned_t5_{name}')
#             print(f"Fine-tuned T5 on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune T5 on dataset: {name} with error: {e}")

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     if 'question' in examples and 'context' in examples:
#         return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)
#     else:
#         raise ValueError("Expected fields 'question' and 'context' not found in examples")

# def fine_tune_qa(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#             qa_training_args = TrainingArguments(
#                 output_dir='./qa_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=3e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             qa_trainer = Trainer(
#                 model=qa_model,
#                 args=qa_training_args,
#                 train_dataset=tokenized_qa_dataset['train'],
#                 eval_dataset=tokenized_qa_dataset['validation'],
#                 data_collator=data_collator,
#             )
#             qa_trainer.train()
#             qa_model.save_pretrained(f'./fine_tuned_qa_{name}')
#             qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{name}')
#             print(f"Fine-tuned BERT for QA on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT for QA on dataset: {name} with error: {e}")

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets(bert_datasets)
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         try:
#             print(f"Loading dataset: {name}")
#             datasets.append((name, load_dataset_with_retries(name, config)))
#             print(f"Loaded dataset: {name} successfully")
#         except Exception as e:
#             print(f"Failed to load dataset: {name} with error: {e}")
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     if 'text' in examples:
#         return bert_tokenizer(examples['text'], padding='max_length', truncation=True)
#     else:
#         raise ValueError("Expected field 'text' not found in examples")

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#             bert_training_args = TrainingArguments(
#                 output_dir='./bert_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=2e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             bert_trainer = Trainer(
#                 model=bert_model,
#                 args=bert_training_args,
#                 train_dataset=tokenized_bert_dataset['train'],
#                 eval_dataset=tokenized_bert_dataset['test']
#             )
#             bert_trainer.train()
#             bert_model.save_pretrained(f'./fine_tuned_bert_{name}')
#             print(f"Fine-tuned BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {name} with error: {e}")

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             sentence_examples = [
#                 InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label'])
#                 for row in dataset['train']
#             ]
#             sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#             sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

#             sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#             sentence_model.save(f'fine_tuned_sentence_bert_{name}')
#             print(f"Fine-tuned Sentence-BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune Sentence-BERT on dataset: {name} with error: {e}")

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     if 'input_text' in examples and 'target_text' in examples:
#         inputs = examples['input_text']
#         targets = examples['target_text']
#         model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#         with t5_tokenizer.as_target_tokenizer():
#             labels = t5_tokenizer(targets, max_length=512, truncation=True)
#         model_inputs['labels'] = labels['input_ids']
#         return model_inputs
#     else:
#         raise ValueError("Expected fields 'input_text' and 'target_text' not found in examples")

# def fine_tune_t5(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#             t5_training_args = TrainingArguments(
#                 output_dir='./t5_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=5e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             t5_trainer = Trainer(
#                 model=t5_model,
#                 args=t5_training_args,
#                 train_dataset=tokenized_t5_dataset['train'],
#                 eval_dataset=tokenized_t5_dataset['validation']
#             )
#             t5_trainer.train()
#             t5_model.save_pretrained(f'fine_tuned_t5_{name}')
#             t5_tokenizer.save_pretrained(f'fine_tuned_t5_{name}')
#             print(f"Fine-tuned T5 on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune T5 on dataset: {name} with error: {e}")

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     if 'question' in examples and 'context' in examples:
#         return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)
#     else:
#         raise ValueError("Expected fields 'question' and 'context' not found in examples")

# def fine_tune_qa(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#             qa_training_args = TrainingArguments(
#                 output_dir='./qa_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=3e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             qa_trainer = Trainer(
#                 model=qa_model,
#                 args=qa_training_args,
#                 train_dataset=tokenized_qa_dataset['train'],
#                 eval_dataset=tokenized_qa_dataset['validation'],
#                 data_collator=data_collator,
#             )
#             qa_trainer.train()
#             qa_model.save_pretrained(f'./fine_tuned_qa_{name}')
#             qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{name}')
#             print(f"Fine-tuned BERT for QA on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT for QA on dataset: {name} with error: {e}")

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# import time
# import pandas as pd
# from transformers import (
#     BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
#     T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, 
#     DataCollatorWithPadding
# )
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     ('stanfordnlp/imdb', None)
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None)
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('squad', None)  # Replace with SQuAD dataset for QA
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10, trust_remote_code=False):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token, trust_remote_code=trust_remote_code)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         try:
#             print(f"Loading dataset: {name}")
#             if name == 'Hello-SimpleAI/HC3':
#                 datasets.append((name, load_dataset_with_retries(name, config, trust_remote_code=True)))
#             else:
#                 datasets.append((name, load_dataset_with_retries(name, config)))
#             print(f"Loaded dataset: {name} successfully")
#         except Exception as e:
#             print(f"Failed to load dataset: {name} with error: {e}")
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     if 'text' in examples:
#         return bert_tokenizer(examples['text'], padding='max_length', truncation=True)
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected field 'text' not found in examples")

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#             bert_training_args = TrainingArguments(
#                 output_dir='./bert_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=2e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             bert_trainer = Trainer(
#                 model=bert_model,
#                 args=bert_training_args,
#                 train_dataset=tokenized_bert_dataset['train'],
#                 eval_dataset=tokenized_bert_dataset['test']
#             )
#             bert_trainer.train()
#             bert_model.save_pretrained(f'./fine_tuned_bert_{name}')
#             print(f"Fine-tuned BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {name} with error: {e}")

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def preprocess_sentence_bert_function(dataset):
#     if 'sentence1' in dataset['train'][0] and 'sentence2' in dataset['train'][0]:
#         sentence_examples = [
#             InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label'])
#             for row in dataset['train']
#         ]
#         return sentence_examples
#     else:
#         print("Examples:", dataset['train'][0])  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'sentence1' and 'sentence2' not found in examples")

# def fine_tune_sentence_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             sentence_examples = preprocess_sentence_bert_function(dataset)
#             sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#             sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

#             sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#             sentence_model.save(f'fine_tuned_sentence_bert_{name}')
#             print(f"Fine-tuned Sentence-BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune Sentence-BERT on dataset: {name} with error: {e}")

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     if 'input_text' in examples and 'target_text' in examples:
#         inputs = examples['input_text']
#         targets = examples['target_text']
#         model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#         with t5_tokenizer.as_target_tokenizer():
#             labels = t5_tokenizer(targets, max_length=512, truncation=True)
#         model_inputs['labels'] = labels['input_ids']
#         return model_inputs
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'input_text' and 'target_text' not found in examples")

# def fine_tune_t5(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#             t5_training_args = TrainingArguments(
#                 output_dir='./t5_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=5e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             t5_trainer = Trainer(
#                 model=t5_model,
#                 args=t5_training_args,
#                 train_dataset=tokenized_t5_dataset['train'],
#                 eval_dataset=tokenized_t5_dataset['validation']
#             )
#             t5_trainer.train()
#             t5_model.save_pretrained(f'./fine_tuned_t5_{name}')
#             print(f"Fine-tuned T5 on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune T5 on dataset: {name} with error: {e}")

# # Fine-tune BERT for Question Answering
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)

# def preprocess_qa_function(examples):
#     if 'question' in examples and 'context' in examples:
#         inputs = qa_tokenizer(
#             examples['question'], examples['context'], 
#             max_length=512, truncation=True, padding="max_length"
#         )
#         return inputs
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'question' and 'context' not found in examples")

# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def fine_tune_qa(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#             qa_training_args = TrainingArguments(
#                 output_dir='./qa_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=3e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             qa_trainer = Trainer(
#                 model=qa_model,
#                 args=qa_training_args,
#                 train_dataset=tokenized_qa_dataset['train'],
#                 eval_dataset=tokenized_qa_dataset['validation']
#             )
#             qa_trainer.train()
#             qa_model.save_pretrained(f'./fine_tuned_qa_{name}')
#             print(f"Fine-tuned QA model on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune QA model on dataset: {name} with error: {e}")

# # Load datasets
# loaded_bert_datasets = load_datasets(bert_datasets)
# loaded_sentence_bert_datasets = load_datasets(sentence_bert_datasets)
# loaded_t5_datasets = load_datasets(t5_datasets)
# loaded_qa_datasets = load_datasets(qa_datasets)

# # Inspect dataset formats
# for name, dataset in loaded_bert_datasets:
#     print(f"Dataset {name} train example: {dataset['train'][0]}")

# for name, dataset in loaded_sentence_bert_datasets:
#     print(f"Dataset {name} train example: {dataset['train'][0]}")

# for name, dataset in loaded_t5_datasets:
#     print(f"Dataset {name} train example: {dataset['train'][0]}")

# for name, dataset in loaded_qa_datasets:
#     print(f"Dataset {name} train example: {dataset['train'][0]}")

# # Fine-tune models
# fine_tune_bert(loaded_bert_datasets)
# fine_tune_sentence_bert(loaded_sentence_bert_datasets)
# fine_tune_t5(loaded_t5_datasets)
# fine_tune_qa(loaded_qa_datasets)

# # Ensure proper device assignment
# print(f"Device for BERT model: {bert_model.device}")
# print(f"Device for Sentence-BERT model: {sentence_model.device}")
# print(f"Device for T5 model: {t5_model.device}")
# print(f"Device for QA model: {qa_model.device}")


# import os
# import time
# import pandas as pd
# from transformers import (
#     BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
#     T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, 
#     DataCollatorWithPadding
# )
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('Hello-SimpleAI/HC3', None),
#     ('McGill-NLP/WebLINX', None),
#     ('rag-datasets/rag-mini-wikipedia', None)
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('HuggingFaceFW/fineweb', None),
#     ('Open-Orca/OpenOrca', None),
#     ('togethercomputer/RedPajama-Data-1T', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('databricks/databricks-dolly-15k', None),
#     ('microsoft/orca-math-word-problems-200k', None)
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         try:
#             print(f"Loading dataset: {name}")
#             datasets.append((name, load_dataset_with_retries(name, config)))
#             print(f"Loaded dataset: {name} successfully")
#         except Exception as e:
#             print(f"Failed to load dataset: {name} with error: {e}")
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     if 'text' in examples:
#         return bert_tokenizer(examples['text'], padding='max_length', truncation=True)
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected field 'text' not found in examples")

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#             bert_training_args = TrainingArguments(
#                 output_dir='./bert_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=2e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             bert_trainer = Trainer(
#                 model=bert_model,
#                 args=bert_training_args,
#                 train_dataset=tokenized_bert_dataset['train'],
#                 eval_dataset=tokenized_bert_dataset['test']
#             )
#             bert_trainer.train()
#             bert_model.save_pretrained(f'./fine_tuned_bert_{name}')
#             print(f"Fine-tuned BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {name} with error: {e}")

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def preprocess_sentence_bert_function(dataset):
#     if 'sentence1' in dataset['train'][0] and 'sentence2' in dataset['train'][0]:
#         sentence_examples = [
#             InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label'])
#             for row in dataset['train']
#         ]
#         return sentence_examples
#     else:
#         print("Examples:", dataset['train'][0])  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'sentence1' and 'sentence2' not found in examples")

# def fine_tune_sentence_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             sentence_examples = preprocess_sentence_bert_function(dataset)
#             sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#             sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

#             sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#             sentence_model.save(f'fine_tuned_sentence_bert_{name}')
#             print(f"Fine-tuned Sentence-BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune Sentence-BERT on dataset: {name} with error: {e}")

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     if 'input_text' in examples and 'target_text' in examples:
#         inputs = examples['input_text']
#         targets = examples['target_text']
#         model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#         with t5_tokenizer.as_target_tokenizer():
#             labels = t5_tokenizer(targets, max_length=512, truncation=True)
#         model_inputs['labels'] = labels['input_ids']
#         return model_inputs
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'input_text' and 'target_text' not found in examples")

# def fine_tune_t5(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#             t5_training_args = TrainingArguments(
#                 output_dir='./t5_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=5e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             t5_trainer = Trainer(
#                 model=t5_model,
#                 args=t5_training_args,
#                 train_dataset=tokenized_t5_dataset['train'],
#                 eval_dataset=tokenized_t5_dataset['validation']
#             )
#             t5_trainer.train()
#             t5_model.save_pretrained(f'./fine_tuned_t5_{name}')
#             print(f"Fine-tuned T5 on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune T5 on dataset: {name} with error: {e}")

# # Fine-tune BERT for Question Answering
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)

# def preprocess_qa_function(examples):
#     if 'question' in examples and 'context' in examples:
#         inputs = qa_tokenizer(
#             examples['question'], examples['context'], 
#             max_length=512, truncation=True, padding="max_length"
#         )
#         return inputs
#     else:
#         print("Examples:", examples)  # Debug print to see what the examples contain
#         raise ValueError("Expected fields 'question' and 'context' not found in examples")

# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def fine_tune_qa(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#             qa_training_args = TrainingArguments(
#                 output_dir='./qa_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=3e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             qa_trainer = Trainer(
#                 model=qa_model,
#                 args=qa_training_args,
#                 train_dataset=tokenized_qa_dataset['train'],
#                 eval_dataset=tokenized_qa_dataset['validation']
#             )
#             qa_trainer.train()
#             qa_model.save_pretrained(f'./fine_tuned_qa_{name}')
#             print(f"Fine-tuned QA model on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune QA model on dataset: {name} with error: {e}")

# # Load datasets
# loaded_bert_datasets = load_datasets([(name, None) for name in bert_datasets])
# loaded_sentence_bert_datasets = load_datasets(sentence_bert_datasets)
# loaded_t5_datasets = load_datasets(t5_datasets)
# loaded_qa_datasets = load_datasets(qa_datasets)

# # Inspect dataset formats
# for name, dataset in loaded_bert_datasets:
#     print(f"Dataset {name} train example: {dataset['train'][0]}")

# for name, dataset in loaded_sentence_bert_datasets:
#     print(f"Dataset {name} train example: {dataset['train'][0]}")

# for name, dataset in loaded_t5_datasets:
#     print(f"Dataset {name} train example: {dataset['train'][0]}")

# for name, dataset in loaded_qa_datasets:
#     print(f"Dataset {name} train example: {dataset['train'][0]}")

# # Fine-tune models
# fine_tune_bert(loaded_bert_datasets)
# fine_tune_sentence_bert(loaded_sentence_bert_datasets)
# fine_tune_t5(loaded_t5_datasets)
# fine_tune_qa(loaded_qa_datasets)

# # Ensure proper device assignment
# print(f"Device for BERT model: {bert_model.device}")
# print(f"Device for Sentence-BERT model: {sentence_model.device}")
# print(f"Device for T5 model: {t5_model.device}")
# print(f"Device for QA model: {qa_model.device}")


# import os
# import time
# import pandas as pd
# from transformers import (
#     BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
#     T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, 
#     DataCollatorWithPadding
# )
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('Hello-SimpleAI/HC3', None),
#     ('McGill-NLP/WebLINX', None),
#     ('rag-datasets/rag-mini-wikipedia', None)
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('HuggingFaceFW/fineweb', None),
#     ('Open-Orca/OpenOrca', None),
#     ('togethercomputer/RedPajama-Data-1T', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('databricks/databricks-dolly-15k', None),
#     ('microsoft/orca-math-word-problems-200k', None)
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         try:
#             print(f"Loading dataset: {name}")
#             datasets.append((name, load_dataset_with_retries(name, config)))
#             print(f"Loaded dataset: {name} successfully")
#         except Exception as e:
#             print(f"Failed to load dataset: {name} with error: {e}")
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     if 'text' in examples:
#         return bert_tokenizer(examples['text'], padding='max_length', truncation=True)
#     else:
#         raise ValueError("Expected field 'text' not found in examples")

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#             bert_training_args = TrainingArguments(
#                 output_dir='./bert_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=2e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             bert_trainer = Trainer(
#                 model=bert_model,
#                 args=bert_training_args,
#                 train_dataset=tokenized_bert_dataset['train'],
#                 eval_dataset=tokenized_bert_dataset['test']
#             )
#             bert_trainer.train()
#             bert_model.save_pretrained(f'./fine_tuned_bert_{name}')
#             print(f"Fine-tuned BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {name} with error: {e}")

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             sentence_examples = [
#                 InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label'])
#                 for row in dataset['train']
#             ]
#             sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#             sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

#             sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#             sentence_model.save(f'fine_tuned_sentence_bert_{name}')
#             print(f"Fine-tuned Sentence-BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune Sentence-BERT on dataset: {name} with error: {e}")

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     if 'input_text' in examples and 'target_text' in examples:
#         inputs = examples['input_text']
#         targets = examples['target_text']
#         model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#         with t5_tokenizer.as_target_tokenizer():
#             labels = t5_tokenizer(targets, max_length=512, truncation=True)
#         model_inputs['labels'] = labels['input_ids']
#         return model_inputs
#     else:
#         raise ValueError("Expected fields 'input_text' and 'target_text' not found in examples")

# def fine_tune_t5(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#             t5_training_args = TrainingArguments(
#                 output_dir='./t5_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=5e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             t5_trainer = Trainer(
#                 model=t5_model,
#                 args=t5_training_args,
#                 train_dataset=tokenized_t5_dataset['train'],
#                 eval_dataset=tokenized_t5_dataset['validation']
#             )
#             t5_trainer.train()
#             t5_model.save_pretrained(f'./fine_tuned_t5_{name}')
#             print(f"Fine-tuned T5 on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune T5 on dataset: {name} with error: {e}")

# # Fine-tune BERT for Question Answering
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)

# def preprocess_qa_function(examples):
#     if 'question' in examples and 'context' in examples:
#         inputs = qa_tokenizer(
#             examples['question'], examples['context'], 
#             max_length=512, truncation=True, padding="max_length"
#         )
#         return inputs
#     else:
#         raise ValueError("Expected fields 'question' and 'context' not found in examples")

# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def fine_tune_qa(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#             qa_training_args = TrainingArguments(
#                 output_dir='./qa_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=3e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             qa_trainer = Trainer(
#                 model=qa_model,
#                 args=qa_training_args,
#                 train_dataset=tokenized_qa_dataset['train'],
#                 eval_dataset=tokenized_qa_dataset['validation']
#             )
#             qa_trainer.train()
#             qa_model.save_pretrained(f'./fine_tuned_qa_{name}')
#             print(f"Fine-tuned QA model on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune QA model on dataset: {name} with error: {e}")

# # Load datasets
# loaded_bert_datasets = load_datasets([(name, None) for name in bert_datasets])
# loaded_sentence_bert_datasets = load_datasets(sentence_bert_datasets)
# loaded_t5_datasets = load_datasets(t5_datasets)
# loaded_qa_datasets = load_datasets(qa_datasets)

# # Fine-tune models
# fine_tune_bert(loaded_bert_datasets)
# fine_tune_sentence_bert(loaded_sentence_bert_datasets)
# fine_tune_t5(loaded_t5_datasets)
# fine_tune_qa(loaded_qa_datasets)





# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         try:
#             print(f"Loading dataset: {name}")
#             datasets.append((name, load_dataset_with_retries(name, config)))
#             print(f"Loaded dataset: {name} successfully")
#         except Exception as e:
#             print(f"Failed to load dataset: {name} with error: {e}")
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     if 'text' in examples:
#         return bert_tokenizer(examples['text'], padding='max_length', truncation=True)
#     else:
#         raise ValueError("Expected field 'text' not found in examples")

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#             bert_training_args = TrainingArguments(
#                 output_dir='./bert_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=2e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             bert_trainer = Trainer(
#                 model=bert_model,
#                 args=bert_training_args,
#                 train_dataset=tokenized_bert_dataset['train'],
#                 eval_dataset=tokenized_bert_dataset['test']
#             )
#             bert_trainer.train()
#             bert_model.save_pretrained(f'./fine_tuned_bert_{name}')
#             print(f"Fine-tuned BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {name} with error: {e}")

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             sentence_examples = [
#                 InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label'])
#                 for row in dataset['train']
#             ]
#             sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#             sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

#             sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#             sentence_model.save(f'fine_tuned_sentence_bert_{name}')
#             print(f"Fine-tuned Sentence-BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune Sentence-BERT on dataset: {name} with error: {e}")

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     if 'input_text' in examples and 'target_text' in examples:
#         inputs = examples['input_text']
#         targets = examples['target_text']
#         model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#         with t5_tokenizer.as_target_tokenizer():
#             labels = t5_tokenizer(targets, max_length=512, truncation=True)
#         model_inputs['labels'] = labels['input_ids']
#         return model_inputs
#     else:
#         raise ValueError("Expected fields 'input_text' and 'target_text' not found in examples")

# def fine_tune_t5(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#             t5_training_args = TrainingArguments(
#                 output_dir='./t5_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=5e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             t5_trainer = Trainer(
#                 model=t5_model,
#                 args=t5_training_args,
#                 train_dataset=tokenized_t5_dataset['train'],
#                 eval_dataset=tokenized_t5_dataset['validation']
#             )
#             t5_trainer.train()
#             t5_model.save_pretrained(f'fine_tuned_t5_{name}')
#             t5_tokenizer.save_pretrained(f'fine_tuned_t5_{name}')
#             print(f"Fine-tuned T5 on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune T5 on dataset: {name} with error: {e}")

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     if 'question' in examples and 'context' in examples:
#         return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)
#     else:
#         raise ValueError("Expected fields 'question' and 'context' not found in examples")

# def fine_tune_qa(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#             qa_training_args = TrainingArguments(
#                 output_dir='./qa_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=3e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )
#             qa_trainer = Trainer(
#                 model=qa_model,
#                 args=qa_training_args,
#                 train_dataset=tokenized_qa_dataset['train'],
#                 eval_dataset=tokenized_qa_dataset['validation'],
#                 data_collator=data_collator,
#             )
#             qa_trainer.train()
#             qa_model.save_pretrained(f'./fine_tuned_qa_{name}')
#             qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{name}')
#             print(f"Fine-tuned BERT for QA on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT for QA on dataset: {name} with error: {e}")

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()







# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         try:
#             print(f"Loading dataset: {name}")
#             datasets.append((name, load_dataset_with_retries(name, config)))
#             print(f"Loaded dataset: {name} successfully")
#         except Exception as e:
#             print(f"Failed to load dataset: {name} with error: {e}")
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Ensure all components are moved to the correct device
# def move_to_device(model, tokenizer, dataset):
#     model.to(device)
#     tokenizer.to(device)
#     for batch in dataset:
#         for key in batch.keys():
#             batch[key] = batch[key].to(device)
#     return model, tokenizer, dataset

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#             tokenized_bert_dataset = tokenized_bert_dataset.map(lambda x: {k: v.to(device) for k, v in x.items()})

#             bert_training_args = TrainingArguments(
#                 output_dir='./bert_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=2e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )

#             bert_trainer = Trainer(
#                 model=bert_model,
#                 args=bert_training_args,
#                 train_dataset=tokenized_bert_dataset['train'],
#                 eval_dataset=tokenized_bert_dataset['test']
#             )
#             bert_trainer.train()
#             bert_model.save_pretrained(f'./fine_tuned_bert_{name}')
#             print(f"Fine-tuned BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {name} with error: {e}")

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#             sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#             sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

#             sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#             sentence_model.save(f'fine_tuned_sentence_bert_{name}')
#             print(f"Fine-tuned Sentence-BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune Sentence-BERT on dataset: {name} with error: {e}")

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def fine_tune_t5(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#             tokenized_t5_dataset = tokenized_t5_dataset.map(lambda x: {k: v.to(device) for k, v in x.items()})

#             t5_training_args = TrainingArguments(
#                 output_dir='./t5_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=5e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )

#             t5_trainer = Trainer(
#                 model=t5_model,
#                 args=t5_training_args,
#                 train_dataset=tokenized_t5_dataset['train'],
#                 eval_dataset=tokenized_t5_dataset['validation']
#             )
#             t5_trainer.train()
#             t5_model.save_pretrained(f'fine_tuned_t5_{name}')
#             t5_tokenizer.save_pretrained(f'fine_tuned_t5_{name}')
#             print(f"Fine-tuned T5 on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune T5 on dataset: {name} with error: {e}")

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# def fine_tune_qa(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#             tokenized_qa_dataset = tokenized_qa_dataset.map(lambda x: {k: v.to(device) for k, v in x.items()})

#             qa_training_args = TrainingArguments(
#                 output_dir='./qa_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=3e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )

#             qa_trainer = Trainer(
#                 model=qa_model,
#                 args=qa_training_args,
#                 train_dataset=tokenized_qa_dataset['train'],
#                 eval_dataset=tokenized_qa_dataset['validation'],
#                 data_collator=data_collator,
#             )
#             qa_trainer.train()
#             qa_model.save_pretrained(f'./fine_tuned_qa_{name}')
#             qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{name}')
#             print(f"Fine-tuned BERT for QA on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT for QA on dataset: {name} with error: {e}")

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         try:
#             print(f"Loading dataset: {name}")
#             datasets.append(load_dataset_with_retries(name, config))
#             print(f"Loaded dataset: {name} successfully")
#         except Exception as e:
#             print(f"Failed to load dataset: {name} with error: {e}")
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for dataset in datasets:
#         try:
#             print(f"Processing dataset: {dataset.info.builder_name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#             tokenized_bert_dataset = tokenized_bert_dataset.map(lambda x: {k: v.to(device) for k, v in x.items()})

#             bert_training_args = TrainingArguments(
#                 output_dir='./bert_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=2e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )

#             bert_trainer = Trainer(
#                 model=bert_model,
#                 args=bert_training_args,
#                 train_dataset=tokenized_bert_dataset['train'],
#                 eval_dataset=tokenized_bert_dataset['test']
#             )
#             bert_trainer.train()
#             bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')
#             print(f"Fine-tuned BERT on dataset: {dataset.info.builder_name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {dataset.info.builder_name} with error: {e}")

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for dataset in datasets:
#         try:
#             print(f"Processing dataset: {dataset.info.builder_name}")
#             sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#             sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#             sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

#             sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#             sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')
#             print(f"Fine-tuned Sentence-BERT on dataset: {dataset.info.builder_name}")
#         except Exception as e:
#             print(f"Failed to fine-tune Sentence-BERT on dataset: {dataset.info.builder_name} with error: {e}")

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def fine_tune_t5(datasets):
#     for dataset in datasets:
#         try:
#             print(f"Processing dataset: {dataset.info.builder_name}")
#             tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#             tokenized_t5_dataset = tokenized_t5_dataset.map(lambda x: {k: v.to(device) for k, v in x.items()})

#             t5_training_args = TrainingArguments(
#                 output_dir='./t5_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=5e-5,
#                 per_device_train_batch_size=8,
#                 per_device_eval_batch_size=8,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )

#             t5_trainer = Trainer(
#                 model=t5_model,
#                 args=t5_training_args,
#                 train_dataset=tokenized_t5_dataset['train'],
#                 eval_dataset=tokenized_t5_dataset['validation']
#             )
#             t5_trainer.train()
#             t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#             t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#             print(f"Fine-tuned T5 on dataset: {dataset.info.builder_name}")
#         except Exception as e:
#             print(f"Failed to fine-tune T5 on dataset: {dataset.info.builder_name} with error: {e}")

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# def fine_tune_qa(datasets):
#     for dataset in datasets:
#         try:
#             print(f"Processing dataset: {dataset.info.builder_name}")
#             tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#             tokenized_qa_dataset = tokenized_qa_dataset.map(lambda x: {k: v.to(device) for k, v in x.items()})

#             qa_training_args = TrainingArguments(
#                 output_dir='./qa_results',
#                 evaluation_strategy="epoch",
#                 learning_rate=3e-5,
#                 per_device_train_batch_size=16,
#                 per_device_eval_batch_size=16,
#                 num_train_epochs=3,
#                 weight_decay=0.01,
#                 use_mps_device=False  # Ensure this parameter is set to False
#             )

#             qa_trainer = Trainer(
#                 model=qa_model,
#                 args=qa_training_args,
#                 train_dataset=tokenized_qa_dataset['train'],
#                 eval_dataset=tokenized_qa_dataset['validation'],
#                 data_collator=data_collator,
#             )
#             qa_trainer.train()
#             qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#             qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#             print(f"Fine-tuned BERT for QA on dataset: {dataset.info.builder_name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT for QA on dataset: {dataset.info.builder_name} with error: {e}")

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset_with_retries(name, config))
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for dataset in datasets:
#         tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#         bert_training_args = TrainingArguments(
#             output_dir='./bert_results',
#             evaluation_strategy="epoch",
#             learning_rate=2e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             use_mps_device=False  # Ensure this parameter is set to False
#         )
#         bert_trainer = Trainer(
#             model=bert_model,
#             args=bert_training_args,
#             train_dataset=tokenized_bert_dataset['train'],
#             eval_dataset=tokenized_bert_dataset['test']
#         )
#         bert_trainer.train()
#         bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for dataset in datasets:
#         sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#         sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#         sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#         sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#         sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def fine_tune_t5(datasets):
#     for dataset in datasets:
#         tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#         t5_training_args = TrainingArguments(
#             output_dir='./t5_results',
#             evaluation_strategy="epoch",
#             learning_rate=5e-5,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             use_mps_device=False  # Ensure this parameter is set to False
#         )
#         t5_trainer = Trainer(
#             model=t5_model,
#             args=t5_training_args,
#             train_dataset=tokenized_t5_dataset['train'],
#             eval_dataset=tokenized_t5_dataset['validation']
#         )
#         t5_trainer.train()
#         t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#         t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# def fine_tune_qa(datasets):
#     for dataset in datasets:
#         tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#         qa_training_args = TrainingArguments(
#             output_dir='./qa_results',
#             evaluation_strategy="epoch",
#             learning_rate=3e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             use_mps_device=False  # Ensure this parameter is set to False
#         )
#         qa_trainer = Trainer(
#             model=qa_model,
#             args=qa_training_args,
#             train_dataset=tokenized_qa_dataset['train'],
#             eval_dataset=tokenized_qa_dataset['validation'],
#             data_collator=data_collator,
#         )
#         qa_trainer.train()
#         qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#         qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset_with_retries(name, config))
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for dataset in datasets:
#         tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#         bert_training_args = TrainingArguments(
#             output_dir='./bert_results',
#             evaluation_strategy="epoch",
#             learning_rate=2e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             use_mps_device=False  # Ensure this parameter is set to False
#         )
#         bert_trainer = Trainer(
#             model=bert_model,
#             args=bert_training_args,
#             train_dataset=tokenized_bert_dataset['train'],
#             eval_dataset=tokenized_bert_dataset['test']
#         )
#         bert_trainer.train()
#         bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for dataset in datasets:
#         sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#         sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#         sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#         sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#         sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def fine_tune_t5(datasets):
#     for dataset in datasets:
#         tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#         t5_training_args = TrainingArguments(
#             output_dir='./t5_results',
#             evaluation_strategy="epoch",
#             learning_rate=5e-5,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             use_mps_device=False  # Ensure this parameter is set to False
#         )
#         t5_trainer = Trainer(
#             model=t5_model,
#             args=t5_training_args,
#             train_dataset=tokenized_t5_dataset['train'],
#             eval_dataset=tokenized_t5_dataset['validation']
#         )
#         t5_trainer.train()
#         t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#         t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# def fine_tune_qa(datasets):
#     for dataset in datasets:
#         tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#         qa_training_args = TrainingArguments(
#             output_dir='./qa_results',
#             evaluation_strategy="epoch",
#             learning_rate=3e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             use_mps_device=False  # Ensure this parameter is set to False
#         )
#         qa_trainer = Trainer(
#             model=qa_model,
#             args=qa_training_args,
#             train_dataset=tokenized_qa_dataset['train'],
#             eval_dataset=tokenized_qa_dataset['validation'],
#             data_collator=data_collator,
#         )
#         qa_trainer.train()
#         qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#         qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset_with_retries(name, config))
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for dataset in datasets:
#         tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#         bert_training_args = TrainingArguments(
#             output_dir='./bert_results',
#             evaluation_strategy="epoch",
#             learning_rate=2e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         bert_trainer = Trainer(
#             model=bert_model,
#             args=bert_training_args,
#             train_dataset=tokenized_bert_dataset['train'],
#             eval_dataset=tokenized_bert_dataset['test']
#         )
#         bert_trainer.train()
#         bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for dataset in datasets:
#         sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#         sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#         sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#         sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#         sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def fine_tune_t5(datasets):
#     for dataset in datasets:
#         tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#         t5_training_args = TrainingArguments(
#             output_dir='./t5_results',
#             evaluation_strategy="epoch",
#             learning_rate=5e-5,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         t5_trainer = Trainer(
#             model=t5_model,
#             args=t5_training_args,
#             train_dataset=tokenized_t5_dataset['train'],
#             eval_dataset=tokenized_t5_dataset['validation']
#         )
#         t5_trainer.train()
#         t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#         t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# def fine_tune_qa(datasets):
#     for dataset in datasets:
#         tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#         qa_training_args = TrainingArguments(
#             output_dir='./qa_results',
#             evaluation_strategy="epoch",
#             learning_rate=3e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         qa_trainer = Trainer(
#             model=qa_model,
#             args=qa_training_args,
#             train_dataset=tokenized_qa_dataset['train'],
#             eval_dataset=tokenized_qa_dataset['validation'],
#             data_collator=data_collator,
#         )
#         qa_trainer.train()
#         qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#         qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset_with_retries(name, config))
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for dataset in datasets:
#         tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#         bert_training_args = TrainingArguments(
#             output_dir='./bert_results',
#             evaluation_strategy="epoch",
#             learning_rate=2e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             use_mps_device=True  # Ensure this parameter is set
#         )
#         bert_trainer = Trainer(
#             model=bert_model,
#             args=bert_training_args,
#             train_dataset=tokenized_bert_dataset['train'],
#             eval_dataset=tokenized_bert_dataset['test']
#         )
#         bert_trainer.train()
#         bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for dataset in datasets:
#         sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#         sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#         sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#         sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#         sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def fine_tune_t5(datasets):
#     for dataset in datasets:
#         tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#         t5_training_args = TrainingArguments(
#             output_dir='./t5_results',
#             evaluation_strategy="epoch",
#             learning_rate=5e-5,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             use_mps_device=True  # Ensure this parameter is set
#         )
#         t5_trainer = Trainer(
#             model=t5_model,
#             args=t5_training_args,
#             train_dataset=tokenized_t5_dataset['train'],
#             eval_dataset=tokenized_t5_dataset['validation']
#         )
#         t5_trainer.train()
#         t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#         t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# def fine_tune_qa(datasets):
#     for dataset in datasets:
#         tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#         qa_training_args = TrainingArguments(
#             output_dir='./qa_results',
#             evaluation_strategy="epoch",
#             learning_rate=3e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             use_mps_device=True  # Ensure this parameter is set
#         )
#         qa_trainer = Trainer(
#             model=qa_model,
#             args=qa_training_args,
#             train_dataset=tokenized_qa_dataset['train'],
#             eval_dataset=tokenized_qa_dataset['validation'],
#             data_collator=data_collator,
#         )
#         qa_trainer.train()
#         qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#         qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()



# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, use_auth_token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset_with_retries(name, config))
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for dataset in datasets:
#         tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#         bert_training_args = TrainingArguments(
#             output_dir='./bert_results',
#             evaluation_strategy="epoch",
#             learning_rate=2e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         bert_trainer = Trainer(
#             model=bert_model,
#             args=bert_training_args,
#             train_dataset=tokenized_bert_dataset['train'],
#             eval_dataset=tokenized_bert_dataset['test']
#         )
#         bert_trainer.train()
#         bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for dataset in datasets:
#         sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#         sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#         sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#         sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#         sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def fine_tune_t5(datasets):
#     for dataset in datasets:
#         tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#         t5_training_args = TrainingArguments(
#             output_dir='./t5_results',
#             evaluation_strategy="epoch",
#             learning_rate=5e-5,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         t5_trainer = Trainer(
#             model=t5_model,
#             args=t5_training_args,
#             train_dataset=tokenized_t5_dataset['train'],
#             eval_dataset=tokenized_t5_dataset['validation']
#         )
#         t5_trainer.train()
#         t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#         t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# def fine_tune_qa(datasets):
#     for dataset in datasets:
#         tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#         qa_training_args = TrainingArguments(
#             output_dir='./qa_results',
#             evaluation_strategy="epoch",
#             learning_rate=3e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         qa_trainer = Trainer(
#             model=qa_model,
#             args=qa_training_args,
#             train_dataset=tokenized_qa_dataset['train'],
#             eval_dataset=tokenized_qa_dataset['validation'],
#             data_collator=data_collator,
#         )
#         qa_trainer.train()
#         qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#         qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup
# device = torch.device("cpu")  # Set device to CPU explicitly

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, use_auth_token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset_with_retries(name, config))
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for dataset in datasets:
#         tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#         bert_training_args = TrainingArguments(
#             output_dir='./bert_results',
#             evaluation_strategy="epoch",
#             learning_rate=2e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         bert_trainer = Trainer(
#             model=bert_model,
#             args=bert_training_args,
#             train_dataset=tokenized_bert_dataset['train'],
#             eval_dataset=tokenized_bert_dataset['test']
#         )
#         bert_trainer.train()
#         bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for dataset in datasets:
#         sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#         sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#         sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#         sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#         sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def fine_tune_t5(datasets):
#     for dataset in datasets:
#         tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#         t5_training_args = TrainingArguments(
#             output_dir='./t5_results',
#             evaluation_strategy="epoch",
#             learning_rate=5e-5,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         t5_trainer = Trainer(
#             model=t5_model,
#             args=t5_training_args,
#             train_dataset=tokenized_t5_dataset['train'],
#             eval_dataset=tokenized_t5_dataset['validation']
#         )
#         t5_trainer.train()
#         t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#         t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# def fine_tune_qa(datasets):
#     for dataset in datasets:
#         tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#         qa_training_args = TrainingArguments(
#             output_dir='./qa_results',
#             evaluation_strategy="epoch",
#             learning_rate=3e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         qa_trainer = Trainer(
#             model=qa_model,
#             args=qa_training_args,
#             train_dataset=tokenized_qa_dataset['train'],
#             eval_dataset=tokenized_qa_dataset['validation'],
#             data_collator=data_collator,
#         )
#         qa_trainer.train()
#         qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#         qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, use_auth_token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset_with_retries(name, config))
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for dataset in datasets:
#         tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#         bert_training_args = TrainingArguments(
#             output_dir='./bert_results',
#             evaluation_strategy="epoch",
#             learning_rate=2e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         bert_trainer = Trainer(
#             model=bert_model,
#             args=bert_training_args,
#             train_dataset=tokenized_bert_dataset['train'],
#             eval_dataset=tokenized_bert_dataset['test']
#         )
#         bert_trainer.train()
#         bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

# def fine_tune_sentence_bert(datasets):
#     for dataset in datasets:
#         sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#         sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#         sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#         sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#         sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def fine_tune_t5(datasets):
#     for dataset in datasets:
#         tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#         t5_training_args = TrainingArguments(
#             output_dir='./t5_results',
#             evaluation_strategy="epoch",
#             learning_rate=5e-5,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         t5_trainer = Trainer(
#             model=t5_model,
#             args=t5_training_args,
#             train_dataset=tokenized_t5_dataset['train'],
#             eval_dataset=tokenized_t5_dataset['validation']
#         )
#         t5_trainer.train()
#         t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#         t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# def fine_tune_qa(datasets):
#     for dataset in datasets:
#         tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#         qa_training_args = TrainingArguments(
#             output_dir='./qa_results',
#             evaluation_strategy="epoch",
#             learning_rate=3e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         qa_trainer = Trainer(
#             model=qa_model,
#             args=qa_training_args,
#             train_dataset=tokenized_qa_dataset['train'],
#             eval_dataset=tokenized_qa_dataset['validation'],
#             data_collator=data_collator,
#         )
#         qa_trainer.train()
#         qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#         qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()



# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, use_auth_token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset_with_retries(name, config))
#     return datasets

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for dataset in datasets:
#         tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#         bert_training_args = TrainingArguments(
#             output_dir='./bert_results',
#             evaluation_strategy="epoch",
#             learning_rate=2e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         bert_trainer = Trainer(
#             model=bert_model,
#             args=bert_training_args,
#             train_dataset=tokenized_bert_dataset['train'],
#             eval_dataset=tokenized_bert_dataset['test']
#         )
#         bert_trainer.train()
#         bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# def fine_tune_sentence_bert(datasets):
#     for dataset in datasets:
#         sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#         sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#         sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#         sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#         sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def fine_tune_t5(datasets):
#     for dataset in datasets:
#         tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#         t5_training_args = TrainingArguments(
#             output_dir='./t5_results',
#             evaluation_strategy="epoch",
#             learning_rate=5e-5,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         t5_trainer = Trainer(
#             model=t5_model,
#             args=t5_training_args,
#             train_dataset=tokenized_t5_dataset['train'],
#             eval_dataset=tokenized_t5_dataset['validation']
#         )
#         t5_trainer.train()
#         t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#         t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# def fine_tune_qa(datasets):
#     for dataset in datasets:
#         tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#         qa_training_args = TrainingArguments(
#             output_dir='./qa_results',
#             evaluation_strategy="epoch",
#             learning_rate=3e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         qa_trainer = Trainer(
#             model=qa_model,
#             args=qa_training_args,
#             train_dataset=tokenized_qa_dataset['train'],
#             eval_dataset=tokenized_qa_dataset['validation'],
#             data_collator=data_collator,
#         )
#         qa_trainer.train()
#         qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#         qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets_loaded = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets_loaded = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets_loaded = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets_loaded)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder
# import time

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Function to load dataset with retries
# def load_dataset_with_retries(name, config, retries=5, delay=10):
#     for attempt in range(retries):
#         try:
#             return load_dataset(name, config, use_auth_token=hf_token)
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed with error: {e}")
#             if attempt < retries - 1:
#                 print(f"Retrying in {delay} seconds...")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Could not load dataset.")
#                 raise

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset_with_retries(name, config))
#     return datasets

# bert_datasets = [(name, None) for name in bert_datasets]
# sentence_bert_datasets = load_datasets(sentence_bert_datasets)
# t5_datasets = load_datasets(t5_datasets)
# qa_datasets = load_datasets(qa_datasets)

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Fine-tune BERT for text classification
# for dataset in bert_datasets:
#     tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#     bert_training_args = TrainingArguments(
#         output_dir='./bert_results',
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     bert_trainer = Trainer(
#         model=bert_model,
#         args=bert_training_args,
#         train_dataset=tokenized_bert_dataset['train'],
#         eval_dataset=tokenized_bert_dataset['test']
#     )
#     bert_trainer.train()
#     bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# for dataset in sentence_bert_datasets:
#     sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#     sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#     sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#     sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#     sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# for dataset in t5_datasets:
#     tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#     t5_training_args = TrainingArguments(
#         output_dir='./t5_results',
#         evaluation_strategy="epoch",
#         learning_rate=5e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     t5_trainer = Trainer(
#         model=t5_model,
#         args=t5_training_args,
#         train_dataset=tokenized_t5_dataset['train'],
#         eval_dataset=tokenized_t5_dataset['validation']
#     )
#     t5_trainer.train()
#     t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#     t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# for dataset in qa_datasets:
#     tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#     qa_training_args = TrainingArguments(
#         output_dir='./qa_results',
#         evaluation_strategy="epoch",
#         learning_rate=3e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     qa_trainer = Trainer(
#         model=qa_model,
#         args=qa_training_args,
#         train_dataset=tokenized_qa_dataset['train'],
#         eval_dataset=tokenized_qa_dataset['validation'],
#         data_collator=data_collator,
#     )
#     qa_trainer.train()
#     qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#     qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')

# # Handle potential download timeouts by splitting the dataset loading and training
# def main():
#     try:
#         # Load datasets with retries
#         print("Loading text classification datasets...")
#         bert_datasets = load_datasets([(name, None) for name in bert_datasets])
        
#         print("Loading sentence similarity datasets...")
#         sentence_bert_datasets = load_datasets(sentence_bert_datasets)
        
#         print("Loading text generation datasets...")
#         t5_datasets = load_datasets(t5_datasets)
        
#         print("Loading question answering datasets...")
#         qa_datasets = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets)
        
#         print("Fine-tuning Sentence-BERT for sentence similarity...")
#         fine_tune_sentence_bert(sentence_bert_datasets)
        
#         print("Fine-tuning T5 for text generation...")
#         fine_tune_t5(t5_datasets)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Functions to fine-tune the models
# def fine_tune_bert(datasets):
#     for dataset in datasets:
#         tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#         bert_training_args = TrainingArguments(
#             output_dir='./bert_results',
#             evaluation_strategy="epoch",
#             learning_rate=2e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         bert_trainer = Trainer(
#             model=bert_model,
#             args=bert_training_args,
#             train_dataset=tokenized_bert_dataset['train'],
#             eval_dataset=tokenized_bert_dataset['test']
#         )
#         bert_trainer.train()
#         bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# def fine_tune_sentence_bert(datasets):
#     for dataset in datasets:
#         sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#         sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#         sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#         sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#         sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# def fine_tune_t5(datasets):
#     for dataset in datasets:
#         tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#         t5_training_args = TrainingArguments(
#             output_dir='./t5_results',
#             evaluation_strategy="epoch",
#             learning_rate=5e-5,
#             per_device_train_batch_size=8,
#             per_device_eval_batch_size=8,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         t5_trainer = Trainer(
#             model=t5_model,
#             args=t5_training_args,
#             train_dataset=tokenized_t5_dataset['train'],
#             eval_dataset=tokenized_t5_dataset['validation']
#         )
#         t5_trainer.train()
#         t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#         t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# def fine_tune_qa(datasets):
#     for dataset in datasets:
#         tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#         qa_training_args = TrainingArguments(
#             output_dir='./qa_results',
#             evaluation_strategy="epoch",
#             learning_rate=3e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
#         qa_trainer = Trainer(
#             model=qa_model,
#             args=qa_training_args,
#             train_dataset=tokenized_qa_dataset['train'],
#             eval_dataset=tokenized_qa_dataset['validation'],
#             data_collator=data_collator,
#         )
#         qa_trainer.train()
#         qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#         qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')

# if __name__ == "__main__":
#     main()


# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets with Configurations
# qa_datasets = [
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('mandarjoshi/trivia_qa', 'rc'),
#     ('google-research-datasets/natural_questions', 'default'),
#     ('microsoft/ms_marco', 'v2.1')
# ]

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset(name, config, use_auth_token=hf_token))
#     return datasets

# bert_datasets = [(name, None) for name in bert_datasets]
# sentence_bert_datasets = load_datasets(sentence_bert_datasets)
# t5_datasets = load_datasets(t5_datasets)
# qa_datasets = load_datasets(qa_datasets)

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Fine-tune BERT for text classification
# for dataset in bert_datasets:
#     tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#     bert_training_args = TrainingArguments(
#         output_dir='./bert_results',
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     bert_trainer = Trainer(
#         model=bert_model,
#         args=bert_training_args,
#         train_dataset=tokenized_bert_dataset['train'],
#         eval_dataset=tokenized_bert_dataset['test']
#     )
#     bert_trainer.train()
#     bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# for dataset in sentence_bert_datasets:
#     sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#     sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#     sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#     sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#     sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# for dataset in t5_datasets:
#     tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#     t5_training_args = TrainingArguments(
#         output_dir='./t5_results',
#         evaluation_strategy="epoch",
#         learning_rate=5e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     t5_trainer = Trainer(
#         model=t5_model,
#         args=t5_training_args,
#         train_dataset=tokenized_t5_dataset['train'],
#         eval_dataset=tokenized_t5_dataset['validation']
#     )
#     t5_trainer.train()
#     t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#     t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# for dataset in qa_datasets:
#     tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#     qa_training_args = TrainingArguments(
#         output_dir='./qa_results',
#         evaluation_strategy="epoch",
#         learning_rate=3e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     qa_trainer = Trainer(
#         model=qa_model,
#         args=qa_training_args,
#         train_dataset=tokenized_qa_dataset['train'],
#         eval_dataset=tokenized_qa_dataset['validation'],
#         data_collator=data_collator,
#     )
#     qa_trainer.train()
#     qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#     qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')


# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', 'wikitext-103-raw-v1')
# ]

# # Question Answering Datasets
# qa_datasets = [
#     'simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted',
#     'mandarjoshi/trivia_qa',
#     'google-research-datasets/natural_questions',
#     'microsoft/ms_marco'
# ]

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset(name, config, use_auth_token=hf_token))
#     return datasets

# bert_datasets = [(name, None) for name in bert_datasets]
# qa_datasets = [(name, None) for name in qa_datasets]

# bert_datasets = load_datasets(bert_datasets)
# sentence_bert_datasets = load_datasets(sentence_bert_datasets)
# t5_datasets = load_datasets(t5_datasets)
# qa_datasets = load_datasets(qa_datasets)

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Fine-tune BERT for text classification
# for dataset in bert_datasets:
#     tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#     bert_training_args = TrainingArguments(
#         output_dir='./bert_results',
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     bert_trainer = Trainer(
#         model=bert_model,
#         args=bert_training_args,
#         train_dataset=tokenized_bert_dataset['train'],
#         eval_dataset=tokenized_bert_dataset['test']
#     )
#     bert_trainer.train()
#     bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# for dataset in sentence_bert_datasets:
#     sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#     sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#     sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#     sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#     sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# for dataset in t5_datasets:
#     tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#     t5_training_args = TrainingArguments(
#         output_dir='./t5_results',
#         evaluation_strategy="epoch",
#         learning_rate=5e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     t5_trainer = Trainer(
#         model=t5_model,
#         args=t5_training_args,
#         train_dataset=tokenized_t5_dataset['train'],
#         eval_dataset=tokenized_t5_dataset['validation']
#     )
#     t5_trainer.train()
#     t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#     t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# for dataset in qa_datasets:
#     tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#     qa_training_args = TrainingArguments(
#         output_dir='./qa_results',
#         evaluation_strategy="epoch",
#         learning_rate=3e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     qa_trainer = Trainer(
#         model=qa_model,
#         args=qa_training_args,
#         train_dataset=tokenized_qa_dataset['train'],
#         eval_dataset=tokenized_qa_dataset['validation'],
#         data_collator=data_collator,
#     )
#     qa_trainer.train()
#     qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#     qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')


# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets with Configurations
# t5_datasets = [
#     ('abisee/cnn_dailymail', '3.0.0'),
#     ('allenai/common_gen', None),
#     ('EdinburghNLP/xsum', None),
#     ('simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted', None),
#     ('Salesforce/wikitext', None)
# ]

# # Question Answering Datasets
# qa_datasets = [
#     'simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted',
#     'mandarjoshi/trivia_qa',
#     'google-research-datasets/natural_questions',
#     'microsoft/ms_marco'
# ]

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         datasets.append(load_dataset(name, config, use_auth_token=hf_token))
#     return datasets

# bert_datasets = [(name, None) for name in bert_datasets]
# qa_datasets = [(name, None) for name in qa_datasets]

# bert_datasets = load_datasets(bert_datasets)
# sentence_bert_datasets = load_datasets(sentence_bert_datasets)
# t5_datasets = load_datasets(t5_datasets)
# qa_datasets = load_datasets(qa_datasets)

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Fine-tune BERT for text classification
# for dataset in bert_datasets:
#     tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#     bert_training_args = TrainingArguments(
#         output_dir='./bert_results',
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     bert_trainer = Trainer(
#         model=bert_model,
#         args=bert_training_args,
#         train_dataset=tokenized_bert_dataset['train'],
#         eval_dataset=tokenized_bert_dataset['test']
#     )
#     bert_trainer.train()
#     bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# for dataset in sentence_bert_datasets:
#     sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#     sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#     sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#     sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#     sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# for dataset in t5_datasets:
#     tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#     t5_training_args = TrainingArguments(
#         output_dir='./t5_results',
#         evaluation_strategy="epoch",
#         learning_rate=5e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     t5_trainer = Trainer(
#         model=t5_model,
#         args=t5_training_args,
#         train_dataset=tokenized_t5_dataset['train'],
#         eval_dataset=tokenized_t5_dataset['validation']
#     )
#     t5_trainer.train()
#     t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#     t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# for dataset in qa_datasets:
#     tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#     qa_training_args = TrainingArguments(
#         output_dir='./qa_results',
#         evaluation_strategy="epoch",
#         learning_rate=3e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     qa_trainer = Trainer(
#         model=qa_model,
#         args=qa_training_args,
#         train_dataset=tokenized_qa_dataset['train'],
#         eval_dataset=tokenized_qa_dataset['validation'],
#         data_collator=data_collator,
#     )
#     qa_trainer.train()
#     qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#     qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')


# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets with Configurations
# sentence_bert_datasets = [
#     ('sentence-transformers/stsb', None),
#     ('AlekseyKorshuk/quora-question-pairs', None),
#     ('SetFit/mrpc', None),
#     ('google-research-datasets/paws-x', 'en')
# ]

# # Text Generation Datasets
# t5_datasets = [
#     'abisee/cnn_dailymail',
#     'allenai/common_gen',
#     'EdinburghNLP/xsum',
#     'simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted',
#     'Salesforce/wikitext'
# ]

# # Question Answering Datasets
# qa_datasets = [
#     'simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted',
#     'mandarjoshi/trivia_qa',
#     'google-research-datasets/natural_questions',
#     'microsoft/ms_marco'
# ]

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name, config in dataset_names:
#         if config:
#             datasets.append(load_dataset(name, config, use_auth_token=True))
#         else:
#             datasets.append(load_dataset(name, use_auth_token=True))
#     return datasets

# bert_datasets = [(name, None) for name in bert_datasets]
# sentence_bert_datasets = load_datasets(sentence_bert_datasets)
# t5_datasets = [(name, None) for name in t5_datasets]
# qa_datasets = [(name, None) for name in qa_datasets]

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Fine-tune BERT for text classification
# for dataset in bert_datasets:
#     tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#     bert_training_args = TrainingArguments(
#         output_dir='./bert_results',
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     bert_trainer = Trainer(
#         model=bert_model,
#         args=bert_training_args,
#         train_dataset=tokenized_bert_dataset['train'],
#         eval_dataset=tokenized_bert_dataset['test']
#     )
#     bert_trainer.train()
#     bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# for dataset in sentence_bert_datasets:
#     sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#     sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#     sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#     sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#     sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# for dataset in t5_datasets:
#     tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#     t5_training_args = TrainingArguments(
#         output_dir='./t5_results',
#         evaluation_strategy="epoch",
#         learning_rate=5e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     t5_trainer = Trainer(
#         model=t5_model,
#         args=t5_training_args,
#         train_dataset=tokenized_t5_dataset['train'],
#         eval_dataset=tokenized_t5_dataset['validation']
#     )
#     t5_trainer.train()
#     t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#     t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# for dataset in qa_datasets:
#     tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#     qa_training_args = TrainingArguments(
#         output_dir='./qa_results',
#         evaluation_strategy="epoch",
#         learning_rate=3e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     qa_trainer = Trainer(
#         model=qa_model,
#         args=qa_training_args,
#         train_dataset=tokenized_qa_dataset['train'],
#         eval_dataset=tokenized_qa_dataset['validation'],
#         data_collator=data_collator,
#     )
#     qa_trainer.train()
#     qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#     qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')


# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, DataCollatorWithPadding
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch
# from huggingface_hub import HfApi, HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Text Classification Datasets
# bert_datasets = [
#     'stanfordnlp/imdb',
#     'fancyzhx/ag_news',
#     'Yelp/yelp_review_full',
#     'fancyzhx/dbpedia_14'
# ]

# # Sentence Similarity Datasets
# sentence_bert_datasets = [
#     'sentence-transformers/stsb',
#     'AlekseyKorshuk/quora-question-pairs',
#     'SetFit/mrpc',
#     'google-research-datasets/paws-x'
# ]

# # Text Generation Datasets
# t5_datasets = [
#     'abisee/cnn_dailymail',
#     'allenai/common_gen',
#     'EdinburghNLP/xsum',
#     'simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted',
#     'Salesforce/wikitext'
# ]

# # Question Answering Datasets
# qa_datasets = [
#     'simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted',
#     'mandarjoshi/trivia_qa',
#     'google-research-datasets/natural_questions',
#     'microsoft/ms_marco'
# ]

# # Load datasets
# def load_datasets(dataset_names):
#     datasets = []
#     for name in dataset_names:
#         datasets.append(load_dataset(name, use_auth_token=True))
#     return datasets

# bert_datasets = load_datasets(bert_datasets)
# sentence_bert_datasets = load_datasets(sentence_bert_datasets)
# t5_datasets = load_datasets(t5_datasets)
# qa_datasets = load_datasets(qa_datasets)

# # Define a function to preprocess the text classification datasets
# def preprocess_bert_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# # Initialize BERT model and tokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Fine-tune BERT for text classification
# for dataset in bert_datasets:
#     tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
#     bert_training_args = TrainingArguments(
#         output_dir='./bert_results',
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     bert_trainer = Trainer(
#         model=bert_model,
#         args=bert_training_args,
#         train_dataset=tokenized_bert_dataset['train'],
#         eval_dataset=tokenized_bert_dataset['test']
#     )
#     bert_trainer.train()
#     bert_model.save_pretrained(f'./fine_tuned_bert_{dataset.info.builder_name}')

# # Fine-tune Sentence-BERT for sentence similarity
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# for dataset in sentence_bert_datasets:
#     sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in dataset['train']]
#     sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
#     sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)
#     sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
#     sentence_model.save(f'fine_tuned_sentence_bert_{dataset.info.builder_name}')

# # Fine-tune T5 for text generation
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# def preprocess_t5_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# for dataset in t5_datasets:
#     tokenized_t5_dataset = dataset.map(preprocess_t5_function, batched=True)
#     t5_training_args = TrainingArguments(
#         output_dir='./t5_results',
#         evaluation_strategy="epoch",
#         learning_rate=5e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     t5_trainer = Trainer(
#         model=t5_model,
#         args=t5_training_args,
#         train_dataset=tokenized_t5_dataset['train'],
#         eval_dataset=tokenized_t5_dataset['validation']
#     )
#     t5_trainer.train()
#     t5_model.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')
#     t5_tokenizer.save_pretrained(f'fine_tuned_t5_{dataset.info.builder_name}')

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
# data_collator = DataCollatorWithPadding(tokenizer=qa_tokenizer)

# def preprocess_qa_function(examples):
#     return qa_tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

# for dataset in qa_datasets:
#     tokenized_qa_dataset = dataset.map(preprocess_qa_function, batched=True)
#     qa_training_args = TrainingArguments(
#         output_dir='./qa_results',
#         evaluation_strategy="epoch",
#         learning_rate=3e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#     qa_trainer = Trainer(
#         model=qa_model,
#         args=qa_training_args,
#         train_dataset=tokenized_qa_dataset['train'],
#         eval_dataset=tokenized_qa_dataset['validation'],
#         data_collator=data_collator,
#     )
#     qa_trainer.train()
#     qa_model.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')
#     qa_tokenizer.save_pretrained(f'./fine_tuned_qa_{dataset.info.builder_name}')



# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch

# # Load BERT dataset
# bert_dataset = load_dataset('imdb')
# # Tokenizer and model for BERT
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def bert_preprocess_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# tokenized_bert_dataset = bert_dataset.map(bert_preprocess_function, batched=True)
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Training arguments for BERT
# bert_training_args = TrainingArguments(
#     output_dir='./bert_results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# # Trainer for BERT
# bert_trainer = Trainer(
#     model=bert_model,
#     args=bert_training_args,
#     train_dataset=tokenized_bert_dataset['train'],
#     eval_dataset=tokenized_bert_dataset['test']
# )

# # Fine-tune BERT
# bert_trainer.train()
# bert_model.save_pretrained('./fine_tuned_bert')

# # Load Sentence-BERT dataset
# sentence_bert_dataset = load_dataset('stsb_multi_mt', 'en')
# sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for row in sentence_bert_dataset['train']]

# # Define Sentence-BERT model and dataloader
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
# sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

# # Fine-tune Sentence-BERT
# sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
# sentence_model.save('fine_tuned_sentence_bert')

# # Load T5 dataset
# t5_dataset = load_dataset('cnn_dailymail', '3.0.0')

# # Tokenizer and model for T5
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# def t5_preprocess_function(examples):
#     inputs = examples['article']
#     targets = examples['highlights']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# tokenized_t5_dataset = t5_dataset.map(t5_preprocess_function, batched=True)

# # Training arguments for T5
# t5_training_args = TrainingArguments(
#     output_dir='./t5_results',
#     evaluation_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# # Trainer for T5
# t5_trainer = Trainer(
#     model=t5_model,
#     args=t5_training_args,
#     train_dataset=tokenized_t5_dataset['train'],
#     eval_dataset=tokenized_t5_dataset['validation']
# )

# # Fine-tune T5
# t5_trainer.train()
# t5_model.save_pretrained('fine_tuned_t5')
# t5_tokenizer.save_pretrained('fine_tuned_t5')


# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration
# from sentence_transformers import SentenceTransformer, InputExample, losses
# from datasets import Dataset, DatasetDict, load_dataset
# from torch.utils.data import DataLoader
# import torch

# # Load BERT dataset
# bert_df = pd.read_csv('bert_data.csv')  # Ensure this CSV file has 'text' and 'label' columns
# bert_dataset = DatasetDict({
#     'train': Dataset.from_pandas(bert_df.sample(frac=0.8, random_state=42)),
#     'eval': Dataset.from_pandas(bert_df.drop(bert_df.sample(frac=0.8, random_state=42).index))
# })

# # Tokenizer and model for BERT
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def bert_preprocess_function(examples):
#     return bert_tokenizer(examples['text'], padding='max_length', truncation=True)

# tokenized_bert_dataset = bert_dataset.map(bert_preprocess_function, batched=True)
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Training arguments for BERT
# bert_training_args = TrainingArguments(
#     output_dir='./bert_results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     use_cpu=True,
# )

# # Trainer for BERT
# bert_trainer = Trainer(
#     model=bert_model,
#     args=bert_training_args,
#     train_dataset=tokenized_bert_dataset['train'],
#     eval_dataset=tokenized_bert_dataset['eval']
# )

# # Fine-tune BERT
# bert_trainer.train()
# bert_model.save_pretrained('./fine_tuned_bert')

# # Load Sentence-BERT dataset
# sentence_bert_df = pd.read_csv('sentence_similarity.csv')  # Ensure this CSV file has 'sentence1', 'sentence2', 'label' columns
# sentence_examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']) for _, row in sentence_bert_df.iterrows()]

# # Define Sentence-BERT model and dataloader
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# sentence_dataloader = DataLoader(sentence_examples, shuffle=True, batch_size=16)
# sentence_loss = losses.CosineSimilarityLoss(model=sentence_model)

# # Fine-tune Sentence-BERT
# sentence_model.fit(train_objectives=[(sentence_dataloader, sentence_loss)], epochs=3, warmup_steps=100)
# sentence_model.save('fine_tuned_sentence_bert')

# # Load T5 dataset
# t5_dataset = load_dataset('t5_data.csv')  # Ensure this dataset has 'input_text' and 'target_text' columns

# # Tokenizer and model for T5
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# def t5_preprocess_function(examples):
#     inputs = examples['input_text']
#     targets = examples['target_text']
#     model_inputs = t5_tokenizer(inputs, max_length=512, truncation=True)
#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=512, truncation=True)
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# tokenized_t5_dataset = t5_dataset.map(t5_preprocess_function, batched=True)

# # Training arguments for T5
# t5_training_args = TrainingArguments(
#     output_dir='./t5_results',
#     evaluation_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# # Trainer for T5
# t5_trainer = Trainer(
#     model=t5_model,
#     args=t5_training_args,
#     train_dataset=tokenized_t5_dataset['train'],
#     eval_dataset=tokenized_t5_dataset['eval']
# )

# # Fine-tune T5
# t5_trainer.train()
# t5_model.save_pretrained('fine_tuned_t5')
# t5_tokenizer.save_pretrained('fine_tuned_t5')


# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset, DatasetDict
# import torch

# # Load dataset
# df = pd.read_csv('math.csv')  # Change this to load the correct CSV file for each subject
# df['text'] = df['question'] + ' ' + df['answer'] + ' ' + df['working_out']
# df = df[['text', 'label']]  # Ensure the DataFrame only has the required columns

# # Split the dataset into train and eval
# train_df = df.sample(frac=0.8, random_state=42)  # 80% for training
# eval_df = df.drop(train_df.index)  # 20% for evaluation

# # Convert to Hugging Face Dataset
# dataset = DatasetDict({
#     'train': Dataset.from_pandas(train_df),
#     'eval': Dataset.from_pandas(eval_df)
# })

# # Tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def preprocess_function(examples):
#     if isinstance(examples['text'], list):
#         examples['text'] = [str(x) for x in examples['text']]
#     return tokenizer(examples['text'], padding='max_length', truncation=True)

# tokenized_datasets = dataset.map(preprocess_function, batched=True)

# # Load model
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Force the model to run on CPU
# device = torch.device("cpu")
# model.to(device)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     use_cpu=True,  # Use CPU
# )

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['eval']
# )

# # Train model
# trainer.train()

# # Save the model
# trainer.save_model('./fine_tuned_model')
