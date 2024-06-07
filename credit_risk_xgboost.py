#%%
'''Notebook for https://medium.com/@loffredo.ds/building-a-predictive-credit-risk-analysis-model-using-xgboost-adf3bf77122a'''
# conda install missingno seaborn xgboost lightgbm scikit-plot imbalanced-learn
#%%
# import libraries
import os
import pandas                 as pd     # data manipulation
import numpy                  as np     # array maniputlation
# import missingno              as msno   # missing data evaluation
import matplotlib.pyplot      as plt    # data visualization
import seaborn                as sns    # statistical data visualization
import difflib
from tqdm import tqdm
import scikitplot             as skplt  # data visualization and machine-learning metrics
# from sklearn.impute           import SimpleImputer    # handling missing values
from sklearn.preprocessing    import LabelEncoder     # categorical data transformation
from sklearn.model_selection  import train_test_split # split into training and test sets
from sklearn.pipeline         import make_pipeline    # pipeline construction
from sklearn.model_selection  import cross_val_score  # performance assessment by cross-validation
from sklearn.preprocessing    import StandardScaler   # data standardization
from imblearn.under_sampling  import RandomUnderSampler     # data balancing
from sklearn.model_selection  import StratifiedKFold        # performance assessment with stratified data
from sklearn.model_selection  import GridSearchCV           # creating grid to evaluate hyperparameters
from sklearn.metrics          import classification_report  # performance report generation
from sklearn.metrics          import roc_auc_score          # performance evaluation by AUC
from sklearn.metrics          import recall_score           # recall performance assessment
from scipy.stats              import norm, zscore                 # statistical analysis (normal distribution)
#
# # data classification models
from sklearn.ensemble         import RandomForestClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.linear_model     import SGDClassifier
from sklearn.svm              import SVC
from sklearn.linear_model     import LogisticRegression
from xgboost                  import XGBClassifier
from lightgbm                 import LGBMClassifier


import warnings                   # notifications
warnings.filterwarnings('ignore') # set notifications to be ignores

# additional settings
plt.style.use('ggplot')
sns.set_style('dark')
SEED = 123
np.random.seed(SEED)
supress_plots = True
# configure the output to show all rows and columns
pd.options.display.max_columns = None

#%%
# import data sets and save them into variables
data_url= "http://dl.dropboxusercontent.com/s/xn2a4kzf0zer0xu/acquisition_train.csv?dl=0"
data_local_path = "data/acquisition_train.csv"
if os.path.exists(data_local_path):
    df_raw = pd.read_csv(data_local_path)
else:
    os.makedirs('data', exist_ok=True)
    df_raw = pd.read_csv(data_url)
    df_raw.to_csv(data_local_path, index=False)
#%%
print(df_raw.head())
print(df_raw.info())
# print missing values ratio
n_rows = df_raw.shape[0]
print((df_raw.isnull().sum() / n_rows).sort_values(ascending=False))
# print unique values
print((df_raw.nunique().sort_values(ascending=False) / n_rows).sort_values(ascending=False))
print(df_raw.isnull().all(axis=0).any())
print(df_raw.isnull().all(axis=1).any())
#%%
print(df_raw.target_default.value_counts())
# get nan count
print(df_raw.target_default.isna().sum())
# df_raw.target_default.fillna('False', inplace=True)
# df_raw.target_default = df_raw.target_default.replace({'True': True, 'False': False})
print(df_raw.target_default.value_counts())
# plot histogram of df_raw.target_default
if not supress_plots:
    plt.figure(figsize=(10, 5))
    sns.countplot(x='target_default', data=df_raw)
    plt.show()
#%%
df_clean = df_raw.copy()
#%%
### REMOVE ATTRIBUTES
# drop columns that contain only one value
single_value_cols = df_clean.columns[df_clean.nunique() == 1]
print(single_value_cols)
df_clean.drop(single_value_cols, axis=1, inplace=True)
# drop unuseful columns
features_to_drop = [
    'ids',  # id, (unique categorical data per row)
    'profile_phone_number',  # phone number, (unique categorical data per row)
    'application_time_applied',  # unrelated to target_default prediction
    'target_fraud',  # 96% missing data
]
df_clean.drop(features_to_drop, axis=1, inplace=True)
X_cols = df_clean.columns.drop('target_default')
# X_df = df_clean[X_cols]
#%%
### REMOVE ROWS WITH MISSING DATA
# drop rows where y is nan
df_clean.dropna(subset=['target_default'], inplace=True)
#%%
### CORRECT BINARY DATA TYPES
print(df_clean.dtypes)
print(df_clean.head())
#%%
df_clean_most_common_values = df_clean.mode().loc[0]

# get unique values of columns with only two unique values
binary_cols = df_clean.columns[df_clean.nunique() == 2]
for col in binary_cols:
    print(df_clean[col].value_counts())
    print('-' * 50)
# print(df_clean[binary_cols].isna().sum() / df_clean.shape[0])

boolean_str_cols = ['target_default', 'facebook_profile']
df_clean.facebook_profile.fillna(df_clean_most_common_values.facebook_profile, inplace=True)
# df_clean[boolean_str_cols].replace({'True': True, 'False': False}, inplace=True)
for col in boolean_str_cols:
    df_clean[col].replace({'True': True, 'False': False}, inplace=True)
#%%
# check email domains, fix possible typos
email_value_counts = df_clean.email.value_counts() / len(df_clean)
print(email_value_counts)
unique_emails = email_value_counts.index
unique_emails_outliers = unique_emails[email_value_counts <= 0.01]
unique_emails_non_outliers = unique_emails[email_value_counts > 0.01]

for outlier in unique_emails_outliers:
    matches = difflib.get_close_matches(outlier, unique_emails_non_outliers, n=1)
    if len(matches) > 0:
        df_clean.email.replace(outlier, matches[0], inplace=True)

# verify that all outliers were fixed
print(df_clean.email.value_counts() / len(df_clean))
#%%
### HANDLE MISSING DATA
categorical_cols_with_nan = [c for c in X_cols[df_clean[X_cols].isna().sum() > 0] if df_clean[c].dtype == 'object']
numeric_cols = df_clean[X_cols].select_dtypes(include=np.number).columns
numeric_cols_with_nan = [c for c in X_cols[df_clean[X_cols].isna().sum() > 0] if c in numeric_cols]
print(df_clean_most_common_values[categorical_cols_with_nan])
print(df_clean_most_common_values[numeric_cols_with_nan])
#%%
# fill na in categorical features with most common value
for col in categorical_cols_with_nan:
    df_clean[col].fillna(df_clean_most_common_values[col], inplace=True)
#%%
# replace missing values with 0
numeric_cols_zero = ['last_amount_borrowed', 'last_borrowed_in_months', 'n_issues']
for c in numeric_cols_zero:
    df_clean[c].fillna(0, inplace=True)
#%%
# replace inf with nan
X_clean = df_clean[X_cols]
inf_cols = X_clean.columns[X_clean.max() == np.inf]
print(inf_cols)
for col in inf_cols:
    df_clean[col].replace(np.inf, np.nan, inplace=True)
#%%
# handle negative values
negative_cols = X_clean[numeric_cols].columns[(X_clean[numeric_cols] < 0).any()]
print(negative_cols)
for col in negative_cols:
    print(f'{col}: {df_clean[col].loc[df_clean[col] < 0].value_counts()}')
# replace -999 in external_data_provider_email_seen_before with 0
df_clean.external_data_provider_email_seen_before = df_clean.external_data_provider_email_seen_before.replace({-999: 0})
#%%
# replace missing numerical values with median
df_clean[numeric_cols] = df_clean[numeric_cols].apply(lambda x: x.fillna(x.median()), axis=0)
#%%
# check missing data
print(df_clean.isnull().sum())
#%%
# use zscore to find outliers
z_scores = zscore(df_clean[numeric_cols], axis=0).abs()
std_limit = 10
cols_outliers = (z_scores > std_limit).any(axis=0)
for col in numeric_cols[cols_outliers]:
    print(df_clean[col].loc[z_scores[col] > std_limit].value_counts())
    if not supress_plots:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df_clean[col])
        plt.show()

# remove rows with income and reported_income outliers
income_outliers = (z_scores[['income', 'reported_income']] > std_limit).any(axis=1)
df_clean = df_clean[~income_outliers]
#%%
#### Prepare data
df_proc = df_clean.copy()
#%%
# extract the categorical attributes
X_cols = df_proc.columns.drop('target_default')
cat_cols = df_proc.select_dtypes(['object', 'bool']).columns

# apply LabelEconder to categorical attributes
for col in cat_cols:
  df_proc[col+'_encoded'] = LabelEncoder().fit_transform(df_proc[col])
  df_proc.drop(col, axis=1, inplace=True)

# check changes
print(df_proc.info())
#%%
# split dataset
df_shuffeled = df_proc.sample(frac=1)
y_col = 'target_default_encoded'
X = df_shuffeled.drop(y_col, axis=1)
y = df_shuffeled[y_col]
print(f'{X.shape=}, {y.shape=}')
#%%
# split training and testing data
## stratify= y (to divide so that the classes have the same proportion)
## random_state so that the result is replicable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, shuffle=True,
                                                    random_state=SEED)
print(f'{X_train.shape=}, {y_train.shape=} {X_test.shape=}, {y_test.shape=}')

#%%
# build model evaluation function
def val_model(X, y, clf, scoring='recall'):
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
    scores = cross_val_score(pipeline, X, y, n_jobs=-1, scoring=scoring)

    print(f"{scoring}: {scores.mean():.4f} (+/- {scores.std():.2f})")
    # return the average of the Recall values obtained in cross-validation
    return scores.mean()

#%%
# instantiate base model
rf = RandomForestClassifier()

# evaluate model performance with the 'val_model' function
precision_baseline = val_model(X_train, y_train, rf, scoring='precision')
recall_baseline = val_model(X_train, y_train, rf, scoring='recall')
# recall is significantly lower than precision, indicating an imbalanced training dataset
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

#%%
# search n_estimators
params_grid = {'n_estimators':range(0,500,50)}
xgb_args = {'learning_rate':0.1, 'random_state':SEED}
best_score, best_params = xgb_hyperparam_search(X_train_rus, y_train_rus, params_grid, **xgb_args)
print(f'best params: {best_score=:.4f}, {best_params=:.4f}')
#%%
# refine search for n_estimators with smaller intervals
param_grid = {'n_estimators':range(75,125,5)}
best_score, best_params = xgb_hyperparam_search(X_train_rus, y_train_rus, param_grid, **xgb_args)
print(f'best params: {best_score=:.4f}, {best_params=:.4f}')
#%%
# search max_depth and min_child_weight
param_grid = {'max_depth': range(1,8,1),
              'min_child_weight': range(1,5,1)}
best_score, best_params = xgb_hyperparam_search(X_train_rus, y_train_rus, param_grid, **xgb_args)
print(f'best params: {best_score=:.4f}, {best_params=:.4f}')
#%%
# TODO: seach learning rate
#%%
### train model with the best hyperparameters
# instantiate the final XGBoost model with the best hyperparameters found
xgb = XGBClassifier(learning_rate=0.1 , n_estimators=85, max_depth=6, min_child_weight=1, gamma=0.1, random_state=SEED)

# train the model with training data
xgb.fit(X_train_rus, y_train_rus)

# standardize test data
X_test = scaler.transform(X_test)
# make predictions with test data
y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
# normalized confusion matrix
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True,
                                    title='Normalized Confusion Matrix',
                                    text_fontsize='large', ax=ax[0])
plt.show()
