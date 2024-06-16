import argparse
from sklearn.metrics import matthews_corrcoef, accuracy_score
import re

parser = argparse.ArgumentParser(description="Parser for language model training", add_help=False)

parser.add_argument('--vector_size', type=int, default=300, help='Embedding size')
parser.add_argument('--min_count', type=int, default=5, help='Ignoring frequency treshold')
parser.add_argument('--workers', type=int, default=4, help='Multicores machines for faster training if cython installed')
parser.add_argument('--epochs', type=int, default=3, help='Epochs')
parser.add_argument('--max_len', type=int, default=2000, help='Maximum length for padded sequences')
parser.add_argument('--n_components', type=int, default=20, help='Number of components for PCA')
parser.add_argument('--n_estimators', type=int, default=50, help='Classifier estimators')
parser.add_argument('--n_iter', type=int, default=100, help='Number of iterations for RandomizedSearchCV')
parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for cross-validation')
parser.add_argument('--scoring', type=int, default=accuracy_score, help='Scoring for nested-cv with XGBoost')
parser.add_argument('--random_state', type=int, default=42, help='Random seed')
parser.add_argument('--eval_metric', type=str, default='logloss', help='Evaluation metric for XGBoost')
parser.add_argument('--output_path', type=str, default='../data/output', help='Path to save output data')
parser.add_argument('--clean_path', type=str, default='../data/cleaned_reports', help='Path to cleaned reports')
parser.add_argument('--cac40_path', type=str, default='../data/CAC40', help='Path to CAC40 data')
parser.add_argument('--year_pattern', type=re.Pattern, default=re.compile(r'_(\d{4})'), help='Pattern to match year in filenames')