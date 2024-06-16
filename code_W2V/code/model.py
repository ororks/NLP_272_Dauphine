import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier

class Word2VecModel:
    def __init__(self, texts, tokenized_texts, vector_size, min_count, workers, epochs):
        self.texts = texts
        self.tokenized_texts = tokenized_texts
        self.vector_size = vector_size
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs 

    def train(self):
        model = Word2Vec(sentences=self.tokenized_texts, vector_size=self.vector_size, min_count=self.min_count, workers=self.workers, epochs=20)
        vocab = {word: idx for idx, word in enumerate(model.wv.index_to_key, 1)}
        vocab["<pad>"] = 0
        vocab["<unk>"] = len(vocab)
        
        embedding_matrix = np.zeros((len(vocab), self.vector_size))
        for word, idx in vocab.items():
            if word in model.wv:
                embedding_matrix[idx] = model.wv[word]

        sequences = []
        for text in self.texts:
            tokens = text.split()
            sequence = [vocab.get(token, vocab["<unk>"]) for token in tokens]
            sequences.append(sequence)

        return vocab, embedding_matrix, sequences


class ClassifierModel:
    def __init__(self, args) -> None:
        self.args = args

    def train(self, X, y):
        param_grid = {
            'max_depth': np.linspace(2, 50, 100, dtype=int), 
            'learning_rate': np.linspace(0.01, 1, 50)
        }
        xgb = XGBClassifier(n_estimators=self.args.n_estimators, 
                            random_state=self.args.random_state, 
                            use_label_encoder=False, 
                            eval_metric=self.args.eval_metric)
        
        return self.nested_cv(xgb, param_grid, 
                              X, 
                              y,
                              n_splits=self.args.n_splits,
                              n_iter=self.args.n_iter, 
                              random_state=self.args.random_state, 
                              scoring=self.args.scoring)


    def nested_cv(self, model, param_grid, X, y, scoring, n_iter, n_splits, random_state):
        scorer = make_scorer(scoring) if scoring == 'mcc' else make_scorer(scoring)
        
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        rs = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_grid, 
            scoring=scorer, 
            refit=True, 
            n_iter=n_iter, 
            cv=inner_cv, 
            random_state=random_state, 
            n_jobs=-1
        )
        rs.fit(X, y)
        
        nested_scores = cross_val_score(rs, X, y, scoring=scorer, cv=outer_cv, n_jobs=-1)
        return rs, nested_scores
