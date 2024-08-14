from datasets import load_dataset

# Text Classification Datasets
bert_datasets = [
    'stanfordnlp/imdb',
    'fancyzhx/ag_news',
    'Yelp/yelp_review_full',
    'fancyzhx/dbpedia_14'
]

# Sentence Similarity Datasets with Configurations
sentence_bert_datasets = [
    ('sentence-transformers/stsb', None),
    ('AlekseyKorshuk/quora-question-pairs', None),
    ('SetFit/mrpc', None),
    ('google-research-datasets/paws-x', 'en')
]

# Text Generation Datasets
t5_datasets = [
    'abisee/cnn_dailymail',
    'allenai/common_gen',
    'EdinburghNLP/xsum',
    'simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted',
    'Salesforce/wikitext'
]

# Question Answering Datasets
qa_datasets = [
    'simpleParadox/SQuAD_v1.1_Du_et_al_2017_formatted',
    'mandarjoshi/trivia_qa',
    'google-research-datasets/natural_questions',
    'microsoft/ms_marco'
]

# Hugging Face token for authentication
hf_token = "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"

# Verify Text Classification Datasets
print("Verifying Text Classification Datasets")
for name in bert_datasets:
    dataset = load_dataset(name, use_auth_token=hf_token)
    print(f"Dataset {name}:")
    print(dataset['train'][0])

# Verify Sentence Similarity Datasets
print("\nVerifying Sentence Similarity Datasets")
for name, config in sentence_bert_datasets:
    dataset = load_dataset(name, config, use_auth_token=hf_token)
    print(f"Dataset {name} with config {config}:")
    print(dataset['train'][0])

# Verify Text Generation Datasets
print("\nVerifying Text Generation Datasets")
for name in t5_datasets:
    dataset = load_dataset(name, use_auth_token=hf_token)
    print(f"Dataset {name}:")
    print(dataset['train'][0])

# Verify Question Answering Datasets
print("\nVerifying Question Answering Datasets")
for name in qa_datasets:
    dataset = load_dataset(name, use_auth_token=hf_token)
    print(f"Dataset {name}:")
    print(dataset['train'][0])
