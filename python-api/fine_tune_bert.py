# import os
# import time
# from transformers import (
#     BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
#     BertForQuestionAnswering, DataCollatorWithPadding
# )
# from datasets import load_dataset
# import torch
# from huggingface_hub import HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup to always use CPU
# device = torch.device("mps" if torch.has_mps else "cpu")
# print(f"Using device: {device}")

# # Text Classification Dataset
# bert_datasets = [
#     ('stanfordnlp/imdb', None)
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
#                 use_mps_device=True  # Set to True for M1 optimization
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
#                 use_mps_device=True  # Set to True for M1 optimization
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
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# import time
# from transformers import (
#     BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
#     BertForQuestionAnswering, DataCollatorWithPadding
# )
# from datasets import load_dataset
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
#                 use_mps_device=True  # Set to True for M1 optimization
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
#                 use_mps_device=True  # Set to True for M1 optimization
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
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
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
#     BertTokenizer, TFBertForSequenceClassification, TFBertForQuestionAnswering, 
#     create_optimizer, DataCollatorWithPadding
# )
# from datasets import load_dataset
# import tensorflow as tf
# from huggingface_hub import HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Device setup for TensorFlow (it automatically uses available GPUs/TPUs if available)
# print(f"Using device: {tf.config.experimental.list_physical_devices()}")

# # Text Classification Dataset
# bert_datasets = [
#     ('stanfordnlp/imdb', None)
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
# bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Fine-tune BERT for text classification
# def fine_tune_bert(datasets):
#     for name, dataset in datasets:
#         try:
#             print(f"Processing dataset: {name}")
#             tokenized_bert_dataset = dataset.map(preprocess_bert_function, batched=True)
            
#             train_dataset = tokenized_bert_dataset['train'].to_tf_dataset(
#                 columns=['input_ids', 'attention_mask', 'token_type_ids'],
#                 label_cols=['label'],
#                 shuffle=True,
#                 batch_size=16
#             )
#             eval_dataset = tokenized_bert_dataset['test'].to_tf_dataset(
#                 columns=['input_ids', 'attention_mask', 'token_type_ids'],
#                 label_cols=['label'],
#                 shuffle=False,
#                 batch_size=16
#             )

#             num_train_steps = len(train_dataset)
#             optimizer, _ = create_optimizer(
#                 init_lr=2e-5,
#                 num_train_steps=num_train_steps,
#                 num_warmup_steps=0
#             )

#             bert_model.compile(optimizer=optimizer, loss=bert_model.compute_loss)

#             bert_model.fit(train_dataset, validation_data=eval_dataset, epochs=3)

#             bert_model.save_pretrained(f'./fine_tuned_bert_{name}')
#             print(f"Fine-tuned BERT on dataset: {name}")
#         except Exception as e:
#             print(f"Failed to fine-tune BERT on dataset: {name} with error: {e}")

# # Fine-tune BERT for question answering
# qa_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# qa_model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')
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
            
#             train_dataset = tokenized_qa_dataset['train'].to_tf_dataset(
#                 columns=['input_ids', 'attention_mask', 'token_type_ids'],
#                 label_cols=['start_positions', 'end_positions'],
#                 shuffle=True,
#                 batch_size=16
#             )
#             eval_dataset = tokenized_qa_dataset['validation'].to_tf_dataset(
#                 columns=['input_ids', 'attention_mask', 'token_type_ids'],
#                 label_cols=['start_positions', 'end_positions'],
#                 shuffle=False,
#                 batch_size=16
#             )

#             num_train_steps = len(train_dataset)
#             optimizer, _ = create_optimizer(
#                 init_lr=3e-5,
#                 num_train_steps=num_train_steps,
#                 num_warmup_steps=0
#             )

#             qa_model.compile(optimizer=optimizer, loss=qa_model.compute_loss)

#             qa_model.fit(train_dataset, validation_data=eval_dataset, epochs=3)

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
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
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
#     BertForQuestionAnswering, DataCollatorWithPadding
# )
# from datasets import load_dataset
# import torch
# from huggingface_hub import HfFolder

# # Save your token in an environment variable or use it directly
# hf_token = "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"
# HfFolder.save_token(hf_token)

# # Check if MPS (Metal Performance Shaders) is available and set the device accordingly
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# print(f"Using device: {device}")

# # Text Classification Dataset
# bert_datasets = [
#     ('stanfordnlp/imdb', None)
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
#                 use_mps_device=device.type == 'mps'  # Set this based on the device type
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
#                 use_mps_device=device.type == 'mps'  # Set this based on the device type
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
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
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
#     BertForQuestionAnswering, DataCollatorWithPadding
# )
# from datasets import load_dataset
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
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
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
#     BertForQuestionAnswering, DataCollatorWithPadding
# )
# from datasets import load_dataset
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
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


# import os
# import time
# from transformers import (
#     BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
#     BertForQuestionAnswering, DataCollatorWithPadding
# )
# from datasets import load_dataset
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
        
#         print("Loading question answering datasets...")
#         qa_datasets_loaded = load_datasets(qa_datasets)
        
#         # Fine-tune models
#         print("Fine-tuning BERT for text classification...")
#         fine_tune_bert(bert_datasets_loaded)
        
#         print("Fine-tuning BERT for question answering...")
#         fine_tune_qa(qa_datasets_loaded)
        
#         print("All models have been fine-tuned successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()
