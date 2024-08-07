import argparse
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from sklearn.multioutput import MultiOutputClassifier


def load_data(file_path):
    return pd.read_csv(file_path)


def train_xgb_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: Dict,
                    fit_args: Dict = None) -> BaseEstimator:
    fit_args = fit_args or {}
    # using XGBClassifier without MultiOutputClassifier due to MultiOutputClassifier nto supporting early stopping with eval_set
    model = xgb.XGBClassifier(**params)
    # model = MultiOutputClassifier(xgb.XGBClassifier(**params))
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_args, verbose=1)
    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train XGBoost model for movie genre classification')
    parser.add_argument('data_path', type=str, help='Path to the processed dataset')
    parser.add_argument('output_model_path', type=str, help='Path to save the trained model')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for train-test split')
    parser.add_argument('--n_estimators', type=int, default=600, help='Number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=6, help='Maximum depth of the tree')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--subsample', type=float, default=0.8, help='Subsample ratio of the training instances')
    parser.add_argument('--eval_metric', type=str, default='logloss', help='Evaluation metric')
    parser.add_argument('--early_stopping_rounds', type=int, default=50, help='Number of rounds without improvements before stopping')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # # Load the processed dataset
    # df = load_data(args.data_path)
    # y_cols = [col for col in df.columns if col.startswith('genres_parsed_')]
    #
    # # Extract features and target
    # y = df[y_cols]
    # X = df.drop(y_cols, axis=1)
    processed_data = np.load(args.data_path)
    X = processed_data['X']
    y = processed_data['y']

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    # Train the model
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'objective': 'binary:logistic',
    }
    if torch.cuda.is_available():
        params.update({'device': 'cuda'})

    fit_args = {
        'eval_metric': args.eval_metric,
        'early_stopping_rounds': args.early_stopping_rounds
    }
    model = train_xgb_model(X_train, y_train, X_val, y_val, params, fit_args)

    # Save the trained model
    joblib.dump(model, args.output_model_path)
    print(f'Model saved to {args.output_model_path}')

    # Evaluate the model
    y_pred = model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("F1 Score:", f1_score(y_val, y_pred, average='weighted'))
    print("Classification Report:\n", classification_report(y_val, y_pred, digits=4))
