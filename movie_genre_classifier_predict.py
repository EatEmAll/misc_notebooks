import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import joblib
import torch
from typing import List, Dict

# set manual seed for reproducibility
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)


# Define the inference function
def predict_movie_genres(movie_data: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    """
    Preprocess the input data and predict movie genres with probabilities.

    Args:
    - movie_data (pd.DataFrame): The input data for a single movie.

    Returns:
    - dict: A dictionary with predicted genres and their probabilities.
    """
    # Convert the input data to a DataFrame
    movie_data = pd.DataFrame(movie_data)
    # Preprocess the input data
    processed_data = preprocessing_pipeline.transform(movie_data)
    # Convert processed_data to numpy array (if it's still a DataFrame)
    if isinstance(processed_data, pd.DataFrame):
        processed_data = processed_data.values
    # Make predictions
    predictions = xgb_model.predict_proba(processed_data)
    # Extract genre names from the model (assuming you saved them during training)
    genre_names = xgb_model.classes_
    # Create a dictionary with genres and their probabilities
    predictions_dict = {genre: prob for genre, prob in zip(genre_names, predictions[0])}
    return predictions_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_pipe_path', type=str, help='Path to the preprocessing pipeline')
    parser.add_argument('model_path', type=str, help='Path to the trained XGBoost model')
    parser.add_argument('--input', type=str, default='', help='json string of input data')
    parser.add_argument('--input_path', type=str, default='', help='path to input data json file')

    args = parser.parse_args()

    # Load the preprocessing pipeline and the trained XGBoost model
    preprocessing_pipeline = joblib.load(args.data_pipe_path)
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
    predicted_genres = predict_movie_genres(input_data)
    print(predicted_genres)
