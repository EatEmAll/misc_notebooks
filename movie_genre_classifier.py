#%%
# TODO: add pip install command for all dependencies
#%%
# import dependencies
import itertools
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import langcodes
from prince import MCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torch
from xgboost import XGBClassifier

# set manual seed for reproducibility
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
#%%
data_path = 'data/movie_data.csv'
df = pd.read_csv(data_path)
#%%
# print columns types, shape, missing values ratio, value counts
print(df.info())
print(f'{df.shape=}')
missing_values_ratio = (df.isnull().sum() / df.shape[0]).sort_values(ascending=False)
print(f'missing values ratio: {print(missing_values_ratio)}')
#%%
# check for rows that have missing values across all features
print(df.isnull().all(axis=0).sum())
#%%
print(df.head())
#%%
# remove irrelevant attributes
# release date doesn't seem to add any useful information for predicting genres
df.drop('release_date', axis=1, inplace=True)
#%%
### There aren't any rows with missing values across all features
#%%
print('unique values:')
str_cols = df.select_dtypes(include='object').columns
for col in str_cols:
    # print number of unique values
    print(f'{col}: {df[col].nunique()}')
#%%
# parse json columns & extract categorical attributes
# even though the keys aren't human readable, the values are
# we'll convert the dicts to lists of values

def parse_json_col(str_):
    return list(eval(str_).values())

json_cols = ['languages', 'genres', 'countries']
# df_raw[json_cols].fillna('{}', inplace=True)
for col in json_cols:
    # treat missing values as empty dict
    df[col] = df[col].fillna('{}')
    df[f'{col}_parsed'] = df[col].astype(str).apply(parse_json_col)
    # drop col
    df.drop(col, axis=1, inplace=True)

parsed_json_cols = [f'{col}_parsed' for col in json_cols]
print(df[parsed_json_cols].head())
#%%
# drop all rows where y is missing
n_rows = df.shape[0]
df.dropna(subset=['genres_parsed'], inplace=True)
print(f'dropped {n_rows - df.shape[0]} rows')
#%%
# print unique values in each of the parsed json columns
for col in parsed_json_cols:
    print('-' * 100)
    print(f'{col}:')
    print(sorted(set(itertools.chain.from_iterable(df[col].values))))
#%%
### some values in languages_parsed need to be merged like "German" and "German Language"
# we'll use langcodes to convert natural language names to language codes
# note: this isn't a perfect solution, langcodes fails to identify some of the languages included in the dataset,  we'll handle them as missing data
# convert to language codes
languages_unique_values = set(itertools.chain.from_iterable(df.languages_parsed.values))
langcode_map = {}
for v in languages_unique_values:
    try:
        langcode_map[v] = langcodes.find(v)
    except LookupError:
        langcode_map[v] = ''

df['langcodes'] = df['languages_parsed'].apply(
    lambda x: [(langcode_map[v].language if isinstance(langcode_map[v], langcodes.Language) else langcode_map[v]) for v in x])
# drop languages_parsed
df.drop('languages_parsed', axis=1, inplace=True)
#%%
# Some values in countries column include languages
# remove" Language" from countries_parsed values
df['countries_parsed'] = df['countries_parsed'].apply(lambda x: [v.replace(' Language', '') for v in x])
#%%
cat_multi_cols = ['langcodes', 'countries_parsed', 'genres_parsed']
for col in cat_multi_cols:
    print(f'{col} unique values: {len(set(itertools.chain.from_iterable(df[col].values)))}')
#%%
#### using one hot encoding will add additional 296 attributes to the dataset, we can use MCA to encode these attributes at a lower dimensionality
# we'll start with one hot encoding and use MCA to reduce the dimensionality
n_features = df.shape[1]
one_hot_cols = []
for col in cat_multi_cols:
    mlb = MultiLabelBinarizer()
    encoded_col = mlb.fit_transform(df[col])
    encoded_df = pd.DataFrame(encoded_col, columns=[f'{col}_{v}' for v in mlb.classes_])
    one_hot_cols.extend(encoded_df.columns)
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(col, axis=1, inplace=True)

print(f'{df.shape=}, {df.shape[1] - n_features} new features added')
#%%
# find best number of components for MCA
n_components_candidates = [int(len(one_hot_cols)*ratio) for ratio in (.5, .7, .8, .9)]
cumulative_eigenvalues = []
for n_components in tqdm(n_components_candidates, desc='fitting MCA'):
    mca = MCA(n_components=n_components)
    mca.fit(df[one_hot_cols])
    cumulative_eigenvalues.append(mca.eigenvalues_.sum())
# plot cumulative eigenvalues
plt.figure(figsize=(10, 5))
plt.plot(n_components_candidates, cumulative_eigenvalues)
plt.show()
#%%
# int(len(one_hot_cols)*.8) gives .95 of the explained variance
n_components = int(len(one_hot_cols)*.8)
mca = MCA(n_components=n_components)
mca_df = mca.fit_transform(df[one_hot_cols])
mca_df.columns = [f'mca_{i}' for i in range(n_components)]
# replace one hot encoded columns with MCA columns
df = pd.concat([df, mca_df], axis=1)
df.drop(one_hot_cols, axis=1, inplace=True)
#%%
# TODO: handle numerical columns
numerical_cols = df.select_dtypes(include='number').columns
print((df[numerical_cols].isnull().sum() / df[numerical_cols].shape[0]))
print(df[numerical_cols].describe())
#%%
### 82% of rows are missing movie_box_office_revenue, we'll drop this column
df.drop('movie_box_office_revenue', axis=1, inplace=True)
# fill feature_length missing values with the median value
df['feature_length'].fillna(df['feature_length'].median(), inplace=True)
#%%
### title and plot_summary attributes are unique per row, therefore it makes more sense to treat them as text features to give them semantic meaning, rather than one-hot encoding them
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_size = model.config.hidden_size

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_embeddings(texts: List[str], batch_size: int):
    all_embeddings = []
    print(f"Total number of records: {len(texts)}")
    print(f"Num batches: {(len(texts) // batch_size) + 1}")

    # Extract embeddings for the texts in batches
    for start_index in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[start_index:start_index + batch_size]

        # Generate tokens and move input tensors to GPU
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract the embeddings. no_grad because the gradient does not need to be computed
        # since this is not a learning task
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the last hidden stated and pool them into a mean vector calculated across the sequence length dimension
        # This will reduce the output vector from [batch_size, sequence_length, hidden_layer_size]
        # to [batch_size, hidden_layer_size] thereby generating the embeddings for all the sequences in the batch
        last_hidden_states = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden_states, dim=1).cpu().tolist()

        # Append to the embeddings list
        all_embeddings.extend(embeddings)

    return all_embeddings
#%%
# replace missing values with empty strings
df['title'].fillna('', inplace=True)
df['plot_summary'].fillna('', inplace=True)
#%%
# convert title and plot_summary into text embeddings
df['title_embeddings'] = get_embeddings(df['title'].values.tolist(), batch_size=128)
df['plot_summary_embeddings'] = get_embeddings(df['plot_summary'].values.tolist(), batch_size=128)
# create attributes to match embeddings size
for i in range(embedding_size):
    df[f'title_embedding_{i}'] = df['title_embeddings'].apply(lambda x: x[i])
    df[f'plot_summary_embedding_{i}'] = df['plot_summary_embeddings'].apply(lambda x: x[i])
# drop title and plot_summary
df.drop(['title', 'plot_summary', 'title_embeddings', 'plot_summary_embeddings'], axis=1, inplace=True)
#%%
print(df.info())
for col in df.columns:
    print(f'{col}: {df[col].dtype}')
print({df[c].dtype for c in df.columns})
#%%
# save the processed data
df.to_csv('data/movie_data_processed.csv', index=False)
#%%
# split dataset
df_shuffled = df.sample(frac=1, random_state=SEED)
y_cols = df_shuffled.columns[df_shuffled.columns.str.startswith('genres_parsed_')]
X = df_shuffled.drop(y_cols, axis=1)
y = df_shuffled[y_cols]
print(f'{X.shape=}, {y.shape=}')
# we'll use train_test_split default 0.25 test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
                                                    # stratify=y,
                                                    shuffle=True,
                                                    random_state=SEED)
print(f'{X_train.shape=}, {y_train.shape=} {X_test.shape=}, {y_test.shape=}')
#%%
# build model evaluation function
def val_model(X, y, clf, scoring, **cross_val_args):
    """
    Performs cross-validation with training data for a given model.

    # Arguments
        X: Data Frame, contains the independent variables.
        y: Series, vector containing the target variable.
        clf:scikit-learn classifier model.
        quite: bool, indicating whether the function should print the results or not.

    # Returns
        float, average of cross-validation scores.
    """

    # convert variables to arrays
    X = np.array(X)
    y = np.array(y)

    # create pipeline
    ## 1. standardize data with StandardScaler
    ## 2. classify the data
    pipeline = make_pipeline(StandardScaler(), clf)

    # model evaluation by cross-validation
    ## according to the Recall value
    scores = cross_val_score(pipeline, X, y, n_jobs=-1, scoring=scoring, **cross_val_args)

    print(f"{scoring}: {scores.mean():.4f} (+/- {scores.std():.2f})")
    # return the average of the Recall values obtained in cross-validation
    return scores.mean()
#%%
# instantiate base model
rf = RandomForestClassifier()

# evaluate model performance with the 'val_model' function
micro_baseline = val_model(X_train, y_train, rf, scoring='recall_micro', cv=3)
#%%
### Data standardization and balancing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train_rus, y_train_rus = RandomUnderSampler(random_state=SEED).fit_resample(X_train, y_train)
print(pd.Series(y_train_rus).value_counts())
#%%
### Compare models
# instantiate the models
rf   = RandomForestClassifier()
knn  = KNeighborsClassifier()
dt   = DecisionTreeClassifier()
sgdc = SGDClassifier()
svc  = SVC()
lr   = LogisticRegression()
xgb  = XGBClassifier()
lgbm = LGBMClassifier()

# create lists to store:
## the classifier model
model = []
## the value of the Recall
recall = []

# create loop to cycle through classification models
for clf in tqdm((rf, knn, dt, sgdc, svc, lr, xgb, lgbm)):

    # identify the classifier
    model.append(clf.__class__.__name__)

    # apply 'val_model' function and store the obtained Recall value
    recall.append(val_model(X_train_rus, y_train_rus, clf, scoring='recall'))

# save the Recall result obtained in each classification model in a variable
results = pd.DataFrame(data=recall, index=model, columns=['Recall'])

# show the models based on the Recall value obtained, from highest to lowest
results.sort_values(by='Recall', ascending=False)
#%%
def xgb_hyperparam_search(X, y, param_grid, **xgb_args):
    # set the learning rate to 0.1 and set the seed
    xgb = XGBClassifier(**xgb_args)
    # set up cross validation with 10 stratified folds
    # shuffle=True to shuffle the data before splitting and setting the seed
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    # configuring the search for cross matches with the XGBoost classifier
    grid_search = GridSearchCV(xgb, param_grid, scoring="recall", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, y)
    return grid_result.best_score_, grid_result.best_params_
