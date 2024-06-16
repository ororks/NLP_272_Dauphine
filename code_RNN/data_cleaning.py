# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:58:45 2024

@author: aikan
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import numpy as np
import pandas as pd
from time import time
from collections import Counter
import nltk
import re
import chardet  # Ensure you have this library installed

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Define the directory paths relative to the notebook location
base_path = Path('..')  # This assumes the notebook is one level deep in the 'code' folder
data_path = base_path / 'data'
cac40_path = data_path / 'CAC40'
processed_path = data_path / 'processed_reports'
clean_path = data_path / 'cleaned_reports'

# Ensure necessary directories exist
for path in [processed_path, clean_path]:
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)
        
"""
# Ensure necessary directories exist
for path in [ngram_path, stats_path]:
    if not path.exists():
        path.mkdir(parents=True)
"""
# Helper function to format time
def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:02.0f}:{m:02.0f}:{s:02.0f}'

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Function to read text file with the correct encoding
def read_text_file(filepath):
    encoding = detect_encoding(filepath)
    try:
        return filepath.read_text(encoding=encoding)
    except UnicodeDecodeError:
        print(f"Could not read file {filepath} with detected encoding {encoding}")
        return None

# Define French stopwords
french_stopwords = set(stopwords.words('french'))

# Initialize counters
vocab = Counter()
total_tokens = 0

start = time()
to_do = 0
done = 0

# Define a regular expression to match the year in filenames
year_pattern = re.compile(r'_(\d{4})')

# Count files to process
for company_folder in cac40_path.iterdir():
    if company_folder.is_dir():
        annual_folder = company_folder / 'Annual'
        if annual_folder.exists():
            for report_file in annual_folder.glob('*.txt'):
                filename = report_file.name
                year_match = year_pattern.search(filename)
                if year_match:
                    year = int(year_match.group(1))
                    if year > 2008:
                        to_do += 1

# Process each file
for company_folder in cac40_path.iterdir():
    if company_folder.is_dir():
        annual_folder = company_folder / 'Annual'
        if annual_folder.exists():
            for report_file in annual_folder.glob('*.txt'):
                filename = report_file.name
                year_match = year_pattern.search(filename)
                if year_match:
                    year = int(year_match.group(1))
                    if year > 2008:
                        clean_file = clean_path / f'{year}_{company_folder.name}.csv'
                        if clean_file.exists():
                            continue
                        
                        text_content = read_text_file(report_file)
                        if text_content is None:
                            print(f"Could not read file: {report_file}")
                            continue
                        
                        text_content = text_content.lower()
                        
                        # Split text content into sentences
                        sentences = sent_tokenize(text_content, language='french')
                        clean_doc = []
                        
                        for s, sentence in enumerate(sentences):
                            words = word_tokenize(sentence, language='french')
                            clean_sentence = [
                                word for word in words if word.isalpha() and word not in french_stopwords
                            ]
                            total_tokens += len(clean_sentence)
                            if len(clean_sentence) > 0:
                                clean_doc.append([s, ' '.join(clean_sentence)])
                        
                        (pd.DataFrame(clean_doc, columns=['sentence', 'text'])
                         .dropna()
                         .to_csv(clean_file, index=False, encoding='utf-8'))  # Ensure UTF-8 encoding here
                        
                        done += 1
                        if done % 10 == 0:
                            duration = time() - start
                            to_go = (to_do - done) * duration / done
                            print(f'{done:>5}\t{format_time(duration)}\t{total_tokens / duration:,.0f}\t{format_time(to_go)}')

print(f'Processing complete: {done} files processed.')