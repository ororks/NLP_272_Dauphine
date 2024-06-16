import warnings
import argparse
import numpy as np
from pathlib import Path
from data_loader import DataProcessor, StockData
from sklearn.model_selection import train_test_split
from model import TransformerModel
from lm_argparser import parser

warnings.filterwarnings('ignore')

# Parse arguments
parser_instance = argparse.ArgumentParser(parents=[parser])
args = parser_instance.parse_args()

# Initialize directories
data_processor = DataProcessor(args)
data_processor.prepare_directories()

# Process annual reports
texts = data_processor.process_files()

# Align sequences with stock returns
stock_data = StockData(args)
aligned_data = stock_data.aligned_data

# Extract texts and labels
X = [item[0] for item in aligned_data]  # Textes des rapports financiers
y = [item[1] for item in aligned_data]  # Signes de rendement

# Diviser les données en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Transformer model
transformer_model = TransformerModel()
print(1)
transformer_model.train(texts=X_train, labels=y_train, output_dir=args.output_path, epochs=args.epochs)
print(2)

# Save model
transformer_model.model.save_pretrained(args.output_path)
print(3)
transformer_model.tokenizer.save_pretrained(args.output_path)
print(4)

# Test model
predictions = transformer_model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy}')
