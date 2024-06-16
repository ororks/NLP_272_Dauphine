import warnings
import subprocess
from pathlib import Path
import json
import numpy as np
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from pathlib import Path
from data_loader import DataProcessor, StockData
from model import Word2VecModel, ClassifierModel
from sklearn.decomposition import PCA
import argparse
from lm_argparser import parser

warnings.filterwarnings('ignore')

# Set randomness in langdetect
DetectorFactory.seed = 0

parser_instance = argparse.ArgumentParser(parents=[parser])
args = parser_instance.parse_args()

# Initialize directories
data_processor = DataProcessor(args)
data_processor.prepare_directories()

# Process annual reports
texts = data_processor.process_files()

# Train Word2Vec model
tokenized_texts = [text.split() for text in texts]

word2vec_model = Word2VecModel(texts=texts, 
                               tokenized_texts=tokenized_texts, 
                               vector_size=args.vector_size, 
                               min_count=args.min_count, 
                               workers=args.workers, 
                               epochs=args.epochs)

vocab, embedding_matrix, sequences = word2vec_model.train()

print(embedding_matrix)

# Save vocabulary and embedding matrix
with open(Path(args.output_path) / 'vocab.json', 'w') as vocab_file:
    json.dump(vocab, vocab_file)
np.save(Path(args.output_path) / 'word_embeddings.npy', embedding_matrix)

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=args.max_len, padding='post')
np.save(Path(args.output_path) / 'padded_sequences.npy', padded_sequences)

# Align sequences with stock returns
stock_data = StockData(args)
aligned_data = stock_data.create_aligned_data()
X = np.array([item[0] for item in aligned_data])
y = np.array([item[1] for item in aligned_data])

np.save(Path(args.output_path) / 'X.npy', X)
np.save(Path(args.output_path) / 'y.npy', y)

# Train classification model for the sign of the return for the month after the annual reports publication
classif_model = ClassifierModel(args)
pca = PCA(n_components=args.n_components)
X_pca = pca.fit_transform(X)


best_xgb, best_xgb_scores = classif_model.train(X_pca, y)
print("Best XGBoost model: ", best_xgb)
print("Best XGBoost scores: ", best_xgb_scores)
