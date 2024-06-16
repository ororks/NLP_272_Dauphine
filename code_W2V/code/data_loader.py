import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
import nltk
import re
import chardet
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from gensim.models import Word2Vec
import json
import yfinance as yf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Fix randomness in langdetect
DetectorFactory.seed = 0

class DataProcessor:
    def __init__(self, args):
        self.args = args
        self.clean_path = Path(args.clean_path)
        self.french_stopwords = set(stopwords.words('french'))
        self.english_stopwords = set(stopwords.words('english'))
        self.vocab = Counter()
        self.total_tokens = 0
        self.to_do = 0
        self.done = 0
    
    def prepare_directories(self):
        base_path = Path('..')
        data_path = base_path / 'data'
        cac40_path = Path(self.args.cac40_path)
        clean_path = Path(self.args.clean_path)
        output_path = Path(self.args.output_path)
        for path in [clean_path, output_path]:
            if not path.exists():
                path.mkdir(parents=True)

    def process_files(self):
        texts = []
        cac40_path = Path(self.args.cac40_path)

        # Try to load preprocessed texts from JSON file
        json_path = Path(self.args.output_path) / 'texts.json'
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as text_file:
                    texts = json.load(text_file)
                    print(f"Loaded {len(texts)} preprocessed texts from {json_path}.")
                    return texts
            except Exception as e:
                print(f"Failed to load texts from {json_path}: {e}")

        # If loading fails, process the files
        print("Iterating over annual reports (around 5min)")
        for company_folder in tqdm(cac40_path.iterdir()):
            if company_folder.is_dir():
                annual_folder = company_folder / 'DR'
                if annual_folder.exists():
                    for report_file in annual_folder.glob('*.txt'):
                        filename = report_file.name
                        year_match = self.args.year_pattern.search(filename)
                        if year_match:
                            year = int(year_match.group(1))
                            if year > 1999:
                                clean_file = self.clean_path / f'{year}_{company_folder.name}.csv'
                                if clean_file.exists():
                                    continue

                                text_content = self.read_text_file(report_file)
                                if text_content is None:
                                    continue

                                text_content = self.replace_special_characters(text_content).lower()

                                try:
                                    language = detect(text_content)
                                except LangDetectException:
                                    continue

                                if language == 'fr':
                                    sentences = sent_tokenize(text_content, language='french')
                                    stop_words = self.french_stopwords
                                    lang = 'french'
                                elif language == 'en':
                                    sentences = sent_tokenize(text_content, language='english')
                                    stop_words = self.english_stopwords
                                    lang = 'english'
                                else:
                                    continue

                                clean_doc = []
                                for s, sentence in enumerate(sentences):
                                    words = word_tokenize(sentence, language=lang)
                                    clean_sentence = [word for word in words if word.isalpha() and word not in stop_words]
                                    self.total_tokens += len(clean_sentence)
                                    if len(clean_sentence) > 0:
                                        clean_doc.append(' '.join(clean_sentence))

                                texts.append(' '.join(clean_doc))
                                
                                (pd.DataFrame(clean_doc, columns=['sentence'])
                                .dropna()
                                .to_csv(clean_file, index=False, encoding='utf-8'))

                                self.done += 1

        # Save the processed texts to JSON file
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=4)
                print(f"Saved {len(texts)} processed texts to {json_path}.")
        except Exception as e:
            print(f"Failed to save texts to {json_path}: {e}")

        return texts
    

    @staticmethod
    def read_text_file(filepath):
        encoding = detect_encoding(filepath)
        try:
            return filepath.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return None

    @staticmethod
    def replace_special_characters(text):
        replacements = {
            "é": "e", "è": "e", "ô": "o",
            "ê": "e", "â": "a"
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text


class StockData:
    def __init__(self, args) -> None:
        self.args = args
        self.stock_data = self.create_aligned_data

    @staticmethod
    def get_return_sign(ticker, year):
        start_date = f'{year}-03-01'
        end_date = f'{year}-04-01'
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data['Return'] = stock_data['Close'].pct_change()
        stock_data = stock_data.dropna()
        monthly_return = stock_data['Return'].sum()
        return 1 if monthly_return > 0 else 0


    def create_aligned_data(self):
        report_files = sorted(Path(self.args.clean_path).glob('*.csv'))
        padded_sequences = np.load(Path(self.args.output_path) / 'padded_sequences.npy')

        cac40_ticker_map = {
            "Accor": "AC.PA", "Airbus": "AIR.PA", "AirLiquide": "AI.PA",
            "ArcelorMittal": "MT.AS", "Atos": "ATO.PA", "BNPParibas": "BNP.PA",
            "Bouygues": "EN.PA", "CapGemini": "CAP.PA", "Danone": "BN.PA",
            "Essilor": "EL.PA", "LafargeHolcim": "HOLN.SW", "Kering": "KER.PA",
            "Lagardère": "MMB.PA", "Legrand": "LR.PA", "Michelin": "ML.PA",
            "Nokia": "NOKIA.HE", "Orange": "ORA.PA", "PSA": "UG.PA",
            "Publicis": "PUB.PA", "Renault": "RNO.PA", "Saint-Gobain": "SGO.PA",
            "Sanofi": "SAN.PA", "SchneiderElectric": "SU.PA", "Sodexo": "SW.PA",
            "STMicroelectronics": "STM.PA", "Technicolor": "TCH.PA", "Valeo": "FR.PA",
            "Veolia": "VIE.PA", "Vinci": "DG.PA", "Vivendi": "VIV.PA"
        }

        aligned_data = []
        for file, seq in zip(report_files, padded_sequences):
            filename = file.stem
            year, company_name = filename.split('_')
            year = int(year)

            if company_name in cac40_ticker_map:
                ticker = cac40_ticker_map[company_name]
                return_sign = self.get_return_sign(ticker, year)
                aligned_data.append((seq, return_sign))

        return aligned_data


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def pad_sequences(sequences):
    max_len = max(len(seq.split()) for seq in sequences)
    padded_sequences = pad_sequences([seq.split() for seq in sequences], maxlen=max_len, padding='post')
    return padded_sequences