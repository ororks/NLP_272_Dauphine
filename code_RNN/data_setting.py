# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:40:50 2024

@author: aikan
"""

import os
import pandas as pd
import random

def extract_unique_words(csv_folder, output_file):
    # Set to store unique words
    unique_words = set()

    # Iterate over all files in the specified directory
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_folder, filename)
            # Read the CSV file
            df = pd.read_csv(file_path)
            # Iterate over the 'text' column
            for text in df['text']:
                # Split text into words and add to the set
                words = text.split()
                unique_words.update(words)

    # Write unique words to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in sorted(unique_words):
            f.write(f"{word}\n")
            

def extract_sentences_to_file(csv_folder, output_file):
    # List to store all sentences
    all_sentences = []

    # Iterate over all files in the specified directory
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_folder, filename)
            # Read the CSV file
            df = pd.read_csv(file_path)
            # Append all sentences in the 'text' column to the list
            all_sentences.extend(df['text'].tolist())

    # Write all sentences to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in all_sentences:
            f.write(f"{sentence}\n")

def split_data(input_file, train_size):
    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Shuffle the lines to ensure random distribution
    random.shuffle(lines)

    # Calculate the split index
    split_index = int(train_size * len(lines))

    # Split the lines into training and testing sets
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]

    # Write the training data to train.txt
    with open('../data/cleaned_reports/train.txt', 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line)

    # Write the testing data to test.txt
    with open('../data/cleaned_reports/test.txt', 'w', encoding='utf-8') as f:
        for line in test_lines:
            f.write(line)


if __name__ == "__main__":
    
    # Define the path to the folder and the output file
    csv_folder = '../data/cleaned_reports'
    output_file = '../data/cleaned_reports/vocab.txt'

    # Call the function
    extract_unique_words(csv_folder, output_file)
    
    # Define the path to the folder and the output file
    csv_folder = '../data/cleaned_reports'
    output_file = '../data/cleaned_reports/valid.txt'

    # Call the function
    extract_sentences_to_file(csv_folder, output_file)
    
    
    # Define the input file and train size
    input_file = '../data/cleaned_reports/valid.txt'
    train_size = 0.6  # For example, 80% train and 20% test

    # Call the function
    split_data(input_file, train_size)
