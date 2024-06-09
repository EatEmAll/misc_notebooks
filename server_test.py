from argparse import ArgumentParser

import requests
import json
import pandas as pd

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to input json file')
    parser.add_argument('--port', type=int, default=8001, help='Port for the server')
    args = parser.parse_args()

    predict_movies_genres_url = f'http://localhost:{args.port}/predict-genres'
    # input_path = 'example_inputs/plot_summary_1.json'
    with open(args.input_path, 'r') as f:
        input_data = json.load(f)

    resp = requests.post(predict_movies_genres_url, json=input_data)
    resp.raise_for_status()
    predicted_genres = resp.json()
    print(pd.Series(predicted_genres).sort_values(ascending=False))
