import json
from argparse import ArgumentParser

import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

from preprocessing_transformers import ParseJsonColumns, ConvertToLangCodes, CleanCountryNames, EncodeMultiLabel, \
    TextEmbeddings

if __name__ == '__main__':
    parser = ArgumentParser()
    # add positional argument for data_path
    parser.add_argument('data_path', type=str, help='Path to the movie data csv/json file')
    parser.add_argument('processed_data_path', type=str, help='Path to the processed movie data CSV file')
    parser.add_argument('--pipeline_path', type=str, help='Path to save the preprocessing pipeline',
                        default='models/preprocessing_pipeline.pkl')
    args = parser.parse_args()
    # Load data
    if args.data_path.endswith('.json'):
        with open(args.data_path) as f:
            rows = list(map(json.loads, f.readlines()))
            df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(args.data_path)

    # Pipeline for preprocessing
    preprocessing_pipeline = Pipeline(steps=[
        ('parse_json', ParseJsonColumns(columns=['languages', 'genres', 'countries'])),
        ('convert_langcodes', ConvertToLangCodes()),
        ('clean_countries', CleanCountryNames()),
        ('multilabel_X', EncodeMultiLabel(['langcodes', 'countries_parsed'], mca_components_ratio=.8)),
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
