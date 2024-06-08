import json
from argparse import ArgumentParser

import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import StandardScaler

from preprocessing import ParseJsonColumns, ConvertToLangCodes, CleanCountryNames, EncodeMultiLabel, \
    TextEmbeddings, MultilabelUnderSampler, FillMissingValues

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
        ('fill_numerical', FillMissingValues(['feature_length'], fill_func='median')),
        ('text_embeddings', TextEmbeddings(['title', 'plot_summary'])),
        ('multilabel_X', EncodeMultiLabel(['langcodes', 'countries_parsed'], mca_components_ratio=.8)),
        ('multilabel_y', EncodeMultiLabel(['genres_parsed'])),
        ('balance_y', MultilabelUnderSampler(['genres_parsed'])),
        ('scaler', StandardScaler()),
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
