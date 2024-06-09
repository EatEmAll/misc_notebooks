"""Preprocess a simplified version of movies_data.csv that contains only the plot_summary and genres columns."""
import json
import os
from argparse import ArgumentParser

import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

from preprocessing import EncodeMultiLabel, TextEmbeddings, MultilabelUnderSampler, ParseJsonColumns, StandardScalerPD

if __name__ == '__main__':

    parser = ArgumentParser()
    # add positional argument for data_path
    parser.add_argument('data_path', type=str, help='Path to the movie data CSV file')
    parser.add_argument('processed_data_path', type=str, help='Path to the processed movie data CSV file')
    parser.add_argument('--pipeline_path', type=str, help='Path to save the preprocessing pipeline', default='models/preprocessing_pipeline.pkl')
    args = parser.parse_args()
    # Load data
    if args.data_path.endswith('.json'):
        with open(args.data_path) as f:
            rows = list(map(json.loads, f.readlines()))
            df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(args.data_path)
    df = df[['plot_summary', 'genres']]

    # Pipeline for preprocessing
    preprocessing_pipeline = Pipeline(steps=[
        ('parse_json', ParseJsonColumns(columns=['genres'])),
        ('text_embeddings', TextEmbeddings(['plot_summary'])),
        ('multilabel_y', EncodeMultiLabel(['genres_parsed'], mca_components_ratio=None)),  # encode y with one-hot without MCA
        ('balance_y', MultilabelUnderSampler(cols_prefix='genres_parsed_')),
        ('scaler', StandardScalerPD(y_cols_prefix='genres_parsed_')),
    ])

    # Fit and transform the data
    df_processed = preprocessing_pipeline.fit_transform(df)
    # Save the processed data
    processed_data_dir = os.path.dirname(args.processed_data_path)
    if processed_data_dir:
        os.makedirs(processed_data_dir, exist_ok=True)
    df_processed.to_csv(args.processed_data_path, index=False)
    # Save the preprocessing pipeline
    data_pipe_dir = os.path.dirname(args.pipeline_path)
    if data_pipe_dir:
        os.makedirs(data_pipe_dir, exist_ok=True)
    joblib.dump(preprocessing_pipeline, args.pipeline_path)

    print(df_processed.info())
    for col in df_processed.columns:
        print(f'{col}: {df_processed[col].dtype}')
    print({df_processed[c].dtype for c in df_processed.columns})
