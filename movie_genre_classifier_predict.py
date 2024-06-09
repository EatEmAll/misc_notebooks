import json
import os.path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import joblib
import torch
from typing import List, Dict

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# set manual seed for reproducibility
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)


# Define the inference function
def predict_movie_genres(xgb_model: BaseEstimator, data_pipe: Pipeline, inputs: List[Dict[str, str]], y_cols: List[str]) \
        ->  Dict[str, Dict[str, float]]:
    """
    Preprocess the input data and predict movie genres with probabilities.

    Args:
    - inputs: batch of inputs structured as a list of dictionaries. Each dictionary contains the data for a single movie.

    Returns:
    - dict: A dictionary with predicted genres and their probabilities.
    """
    # Convert the input data to a DataFrame
    inputs = pd.DataFrame(inputs)
    # Preprocess the input data
    processed_data = data_pipe.transform(inputs)
    # Convert processed_data to numpy array (if it's still a DataFrame)
    if isinstance(processed_data, pd.DataFrame):
        processed_data = processed_data.values
    # Make predictions
    preds = xgb_model.predict_proba(processed_data)
    # preds_bool = preds.round().astype(bool)

    # Create a dictionary with genres and their probabilities
    predictions_dict = {genre: prob for genre, prob in zip(y_cols, preds[0])}
    return predictions_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_pipe_path', type=str, help='Path to the preprocessing pipeline')
    parser.add_argument('model_path', type=str, help='Path to the trained XGBoost model')
    parser.add_argument('--input', type=str, default='', help='json string of input data')
    parser.add_argument('--input_path', type=str, default='', help='path to input data json file')

    args = parser.parse_args()

    # Load the preprocessing pipeline and the trained XGBoost model
    _, data_pipe = joblib.load(args.data_pipe_path)
    data_pipe_ext = os.path.basename(args.data_pipe_path).rsplit('.', 1)[1]
    y_cols_path = args.data_pipe_path.replace(f'.{data_pipe_ext}', '_y_cols.json')
    with open(y_cols_path, 'r') as f:
        y_cols = json.load(f)
    xgb_model = joblib.load(args.model_path)
    if args.input:
        input_data = json.loads(args.input)
    elif args.input_path:
        with open(args.input_path, 'r') as f:
            input_data = json.load(f)
    else:
        raise ValueError("Either --input or --input_path must be provided.")

    # # Example usage
    # # Assuming `example_movie_data` is a DataFrame with a single movie's data
    # example_movie_data = pd.DataFrame({
    #     'title': ['Example Title'],
    #     'plot_summary': ['Example plot summary'],
    #     'languages': ['{"1": "English"}'],
    #     'genres': ['{"1": "Action"}'],
    #     'countries': ['{"1": "USA"}'],
    #     # Add other necessary columns with appropriate values
    # })

    # Preprocess and predict
    predicted_genres = predict_movie_genres(xgb_model, data_pipe, input_data, y_cols)
    predicted_genres = pd.Series(predicted_genres)
    print(predicted_genres.sort_values(ascending=False))
