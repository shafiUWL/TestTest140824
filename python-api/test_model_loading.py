import os

# Set environment variable to avoid a known issue with KMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

print("Model and tokenizer loaded successfully")
