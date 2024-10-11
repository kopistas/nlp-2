# data_utils.py

import pandas as pd
import numpy as np
import nltk
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import torch

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, index_col=None)
    data['Log1pSalary'] = np.log1p(data['SalaryNormalized']).astype('float32')

    text_columns = ["Title", "FullDescription"]
    categorical_columns = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]

    data[categorical_columns] = data[categorical_columns].fillna('NaN')  # Cast missing values to string "NaN"

    tokenizer = nltk.tokenize.WordPunctTokenizer()
    def preprocess_text(text):
        if isinstance(text, str):
            tokens = tokenizer.tokenize(text.lower())  
            return ' '.join(tokens)
        return ''
    
    for col in text_columns:
        data[col] = data[col].apply(preprocess_text)

    # Count how many times each token occurs in both "Title" and "FullDescription" in total
    token_counts = Counter()
    for col in text_columns:
        for text in data[col]:
            tokens = text.split() 
            token_counts.update(tokens)

    # Define special tokens
    UNK = "<UNK>"
    PAD = "<PAD>"

    # Build vocabulary of top tokens
    tokens, counts = zip(*token_counts.most_common())
    tokens = list(tokens)
    tokens = [PAD, UNK] + tokens  # Prepend PAD and UNK tokens

    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    UNK_IX = token_to_id[UNK]
    PAD_IX = token_to_id[PAD]

    # Convert text to sequences of token ids
    def text_to_sequence(text):
        return [token_to_id.get(token, UNK_IX) for token in text.split()]
    
    for col in text_columns:
        data[col] = data[col].apply(text_to_sequence)

    # Process categorical features
    top_companies, top_counts = zip(*Counter(data['Company']).most_common(1000))
    recognized_companies = set(top_companies)
    data["Company"] = data["Company"].apply(lambda comp: comp if comp in recognized_companies else "Other")
    
    categorical_vectorizer = DictVectorizer(dtype=np.float32, sparse=False)
    categorical_vectorizer.fit(data[categorical_columns].to_dict(orient='records'))

    return data, token_to_id, categorical_vectorizer, categorical_columns, UNK_IX, PAD_IX

def prepare_data_splits(data):
    print("Splitting data...")
    from sklearn.model_selection import train_test_split

    data_train, data_val = train_test_split(data, test_size=0.2, random_state=42)
    data_train.index = range(len(data_train))
    data_val.index = range(len(data_val))

    print("Train size =", len(data_train))
    print("Validation size =", len(data_val))

    return data_train, data_val

def as_matrix(sequences, max_len=None, PAD_IX=0):
    """Convert list of sequences into 2D numpy.array of shape [len(sequences), max_len], with padding."""
    if max_len is None:
        max_len = max(map(len, sequences))
    matrix = np.full((len(sequences), max_len), PAD_IX, dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        matrix[i, :length] = seq[:length]
    return matrix

def load_pretrained_embeddings(token_to_id, embedding_name='glove-wiki-gigaword-300'):
    import gensim.downloader as api
    embeddings = api.load(embedding_name)
    embedding_dim = embeddings.vector_size
    embedding_matrix = np.zeros((len(token_to_id), embedding_dim))
    for word, idx in token_to_id.items():
        if word in embeddings:
            embedding_vector = embeddings[word]
            embedding_matrix[idx] = embedding_vector
        else:
            # Random initialization for unknown words
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return torch.tensor(embedding_matrix, dtype=torch.float32)
