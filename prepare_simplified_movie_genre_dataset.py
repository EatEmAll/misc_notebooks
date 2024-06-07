"""Preprocess a simplified version of movies_data.csv that contains only the plot_summary and genres columns."""
from argparse import ArgumentParser

import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

from preprocessing_transformers import EncodeMultiLabel, TextEmbeddings


if __name__ == '__main__':

    parser = ArgumentParser()
    # add positional argument for data_path
    parser.add_argument('data_path', type=str, help='Path to the movie data CSV file')
    parser.add_argument('processed_data_path', type=str, help='Path to the processed movie data CSV file')
    parser.add_argument('--pipeline_path', type=str, help='Path to save the preprocessing pipeline', default='models/preprocessing_pipeline.pkl')
    args = parser.parse_args()
    # Load data
    df = pd.read_csv(args.data_path)
    df = df[['plot_summary', 'genres']]

    # Pipeline for preprocessing
    preprocessing_pipeline = Pipeline(steps=[
        ('multilabel_y', EncodeMultiLabel(['genres_parsed'])),
        ('text_embeddings', TextEmbeddings())
    ])

    # Fit and transform the data
    df_processed = preprocessing_pipeline.fit_transform(df)
    # Save the processed data
    df_processed.to_csv(args.processed_data_path, index=False)
    # Save the preprocessing pipeline
    joblib.dump(preprocessing_pipeline, args.pipeline_path)

    print(df_processed.info())
    for col in df_processed.columns:
        print(f'{col}: {df_processed[col].dtype}')
    print({df_processed[c].dtype for c in df_processed.columns})