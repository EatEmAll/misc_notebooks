import itertools
from typing import List

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
import langcodes
from prince import MCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch


class ParseJsonColumns(BaseEstimator, TransformerMixin):
    """Custom transformer for parsing JSON columns"""
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.columns:
            X[col] = X[col].fillna('{}')
            X[f'{col}_parsed'] = X[col].astype(str).apply(lambda x: list(eval(x).values()))
            X.drop(col, axis=1, inplace=True)
        return X


class ConvertToLangCodes(BaseEstimator, TransformerMixin):
    """Custom transformer for handling language codes"""
    def fit(self, X: pd.DataFrame, y=None):
        languages_unique_values = set(itertools.chain.from_iterable(X['languages_parsed'].values))
        self.langcode_map = {}
        for v in languages_unique_values:
            try:
                self.langcode_map[v] = langcodes.find(v)
            except LookupError:
                self.langcode_map[v] = ''
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X['langcodes'] = X['languages_parsed'].apply(
            lambda x: [(self.langcode_map[v].language if isinstance(self.langcode_map[v], langcodes.Language) else self.langcode_map[v]) for v in x])
        X.drop('languages_parsed', axis=1, inplace=True)
        return X


class CleanCountryNames(BaseEstimator, TransformerMixin):
    """Custom transformer for handling countries"""
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X['countries_parsed'] = X['countries_parsed'].apply(lambda x: [v.replace(' Language', '') for v in x])
        return X


class EncodeMultiLabel(BaseEstimator, TransformerMixin):
    """
    Encode multilabel feature with one-hot apply dimensionality reduction using MCA.
    If mca_components_ratio is None, only one-hot encoding is applied.
    """
    def __init__(self, columns: List[str], mca_components_ratio: float = None):
        self.columns = columns
        self.mca_components_ratio = mca_components_ratio

    def fit(self, X: pd.DataFrame, y=None):
        self.mlb = MultiLabelBinarizer()
        self.one_hot_encoded_cols = []
        for col in self.columns:
            encoded_col = self.mlb.fit_transform(X[col])
            encoded_df = pd.DataFrame(encoded_col, columns=[f'{col}_{v}' for v in self.mlb.classes_])
            self.one_hot_encoded_cols.extend(encoded_df.columns)

        if self.mca_components_ratio is not None:
            self.n_components = int(len(self.one_hot_encoded_cols) * self.mca_components_ratio)
            self.mca = MCA(n_components=self.n_components)
            self.mca.fit(X[self.one_hot_encoded_cols])
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for col in self.columns:
            encoded_col = self.mlb.transform(X[col])
            encoded_df = pd.DataFrame(encoded_col, columns=[f'{col}_{v}' for v in self.mlb.classes_])
            X = pd.concat([X, encoded_df], axis=1)
            X.drop(col, axis=1, inplace=True)

        if self.mca_components_ratio is not None:
            mca_df = self.mca.transform(X[self.one_hot_encoded_cols])
            mca_df = pd.DataFrame(mca_df, columns=[f'mca_{i}' for i in range(self.n_components)])
            X = pd.concat([X, mca_df], axis=1)
            X.drop(self.one_hot_encoded_cols, axis=1, inplace=True)
        return X


class TextEmbeddings(BaseEstimator, TransformerMixin):
    """Custom transformer for text embeddings"""
    def __init__(self, columns: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 128):
        self.columns = columns
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_size = self.model.config.hidden_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        for col in self.columns:
            X[col].fillna('', inplace=True)
            X[f'{col}_embeddings'] = self.get_embeddings(X[col].values.tolist())
            for i in range(self.embedding_size):
                X[f'{col}_embedding_{i}'] = X[f'{col}_embeddings'].apply(lambda x: x[i])
            X.drop([col, f'{col}_embeddings'], axis=1, inplace=True)
        return X

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for start_index in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = texts[start_index:start_index + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embeddings = torch.mean(last_hidden_states, dim=1).cpu().tolist()
            all_embeddings.extend(embeddings)
        return all_embeddings
