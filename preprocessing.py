import itertools
from typing import List, Any

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
import langcodes
from prince import MCA
from skmultilearn.problem_transform import LabelPowerset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch


class BaseTransform(BaseEstimator, TransformerMixin):
    """Base class for custom transformers"""
    def fit(self, X: pd.DataFrame, y=None):
        print(f'Applying fit on {self.__class__.__name__}')
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        print(f'Applying transform on {self.__class__.__name__}')
        return X


class ParseJsonColumns(BaseTransform):
    """Custom transformer for parsing JSON columns"""
    def __init__(self, columns: List[str]):
        self.columns = columns

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        super().transform(X, y)
        for col in self.columns:
            X[col] = X[col].fillna('{}')
            X[f'{col}_parsed'] = X[col].astype(str).apply(lambda x: list(eval(x).values()))
            X.drop(col, axis=1, inplace=True)
        return X


class ConvertToLangCodes(BaseTransform):
    """Custom transformer for handling language codes"""
    def fit(self, X: pd.DataFrame, y=None):
        super().fit(X, y)
        languages_unique_values = set(itertools.chain.from_iterable(X['languages_parsed'].values))
        self.langcode_map = {}
        for v in languages_unique_values:
            try:
                self.langcode_map[v] = langcodes.find(v)
            except LookupError:
                self.langcode_map[v] = ''
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        super().transform(X, y)
        X['langcodes'] = X['languages_parsed'].apply(
            lambda x: [(self.langcode_map[v].language if isinstance(self.langcode_map[v], langcodes.Language) else self.langcode_map[v]) for v in x])
        X.drop('languages_parsed', axis=1, inplace=True)
        return X


class CleanCountryNames(BaseTransform):
    """Custom transformer for handling countries"""
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        super().transform(X, y)
        X['countries_parsed'] = X['countries_parsed'].apply(lambda x: [v.replace(' Language', '') for v in x])
        return X


class EncodeMultiLabel(BaseTransform):
    """
    Encode multilabel feature with one-hot apply dimensionality reduction using MCA.
    If mca_components_ratio is None, only one-hot encoding is applied.
    """
    def __init__(self, columns: List[str], mca_components_ratio: float = None):
        self.columns = columns
        self.mca_components_ratio = mca_components_ratio

    def fit(self, X: pd.DataFrame, y=None):
        super().fit(X, y)
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
        super().transform(X, y)
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


class MultilabelUnderSampler(BaseTransform):
    """Custom transformer for balancing multilabel data using under-sampling"""
    def __init__(self, columns: List[str] = (), cols_prefix: str = ''):
        if not (len(columns) > 0 or cols_prefix):
            raise ValueError("At least one of columns or cols_prefix must be provided.")
        self.columns = columns
        self.cols_prefix = cols_prefix
        self.lp = LabelPowerset()
        self.rus = RandomUnderSampler()

    def fit(self, X: pd.DataFrame, y=None):
        super().fit(X, y)
        if len(self.columns) == 0:
            self.columns = [col for col in X.columns if col.startswith(self.cols_prefix)]
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        super().transform(X, y)
        y = X[self.columns]
        X = X.drop(self.columns, axis=1)
        yt = self.lp.transform(y)
        X_resampled, y_resampled = self.rus.fit_resample(X, yt)
        y_resampled = self.lp.inverse_transform(y_resampled).toarray()
        y_resampled = pd.DataFrame(y_resampled, columns=self.columns, index=X_resampled.index, dtype=int)
        return pd.concat([X_resampled, y_resampled], axis=1)


class TextEmbeddings(BaseTransform):
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

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        super().transform(X, y)
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


class FillMissingValues(BaseTransform):
    """Custom transformer for filling missing values"""
    def __init__(self, columns: List[str], fill_value: Any = None, fill_func: str = None):
        self.columns = columns
        if fill_value is None and fill_func is None:
            raise ValueError("Either fill_value or fill_func must be provided.")
        self.fill_value = fill_value
        self.fill_func = fill_func

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        super().transform(X, y)
        for col in self.columns:
            fill_value = getattr(X[col], self.fill_func)() if self.fill_value is None else self.fill_value
            X[col].fillna(fill_value, inplace=True)
        return X


class StandardScalerPD(BaseTransform):
    """Custom transformer for standard scaling that returns a pandas DataFrame"""
    def fit(self, X: pd.DataFrame, y=None):
        super().fit(X, y)
        self.mean_ = X.mean()
        self.std_ = X.std()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        super().transform(X, y)
        X = (X - self.mean_) / self.std_
        return X
