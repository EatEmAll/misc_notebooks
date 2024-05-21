#%%
'''Notebook for https://medium.com/@loffredo.ds/building-a-predictive-credit-risk-analysis-model-using-xgboost-adf3bf77122a'''
# conda install missingno seaborn xgboost lightgbm scikit-plot imbalanced-learn
#%%
# import libraries
import pandas                 as pd     # data manipulation
import numpy                  as np     # array maniputlation
# import missingno              as msno   # missing data evaluation
import matplotlib.pyplot      as plt    # data visualization
import seaborn                as sns    # statistical data visualization
# import scikitplot             as skplt  # data visualization and machine-learning metrics
# from sklearn.impute           import SimpleImputer    # handling missing values
# from sklearn.preprocessing    import LabelEncoder     # categorical data transformation
# from sklearn.model_selection  import train_test_split # split into training and test sets
# from sklearn.pipeline         import make_pipeline    # pipeline construction
# from sklearn.model_selection  import cross_val_score  # performance assessment by cross-validation
# from sklearn.preprocessing    import StandardScaler   # data standardization
# from imblearn.under_sampling  import RandomUnderSampler     # data balancing
# from sklearn.model_selection  import StratifiedKFold        # performance assessment with stratified data
# from sklearn.model_selection  import GridSearchCV           # creating grid to evaluate hyperparameters
# from sklearn.metrics          import classification_report  # performance report generation
# from sklearn.metrics          import roc_auc_score          # performance evaluation by AUC
# from sklearn.metrics          import recall_score           # recall performance assessment
# from scipy.stats              import norm                   # statistical analysis (normal distribution)
#
# # data classification models
# from sklearn.ensemble         import RandomForestClassifier
# from sklearn.tree             import DecisionTreeClassifier
# from sklearn.neighbors        import KNeighborsClassifier
# from sklearn.linear_model     import SGDClassifier
# from sklearn.svm              import SVC
# from sklearn.linear_model     import LogisticRegression
# from xgboost                  import XGBClassifier
# from lightgbm                 import LGBMClassifier


import warnings                   # notifications
warnings.filterwarnings('ignore') # set notifications to be ignores

# additional settings
plt.style.use('ggplot')
sns.set_style('dark')
np.random.seed(123)

# configure the output to show all rows and columns
pd.options.display.max_columns = None

#%%
# import data sets and save them into variables
# data_path = "http://dl.dropboxusercontent.com/s/xn2a4kzf0zer0xu/acquisition_train.csv?dl=0"
data_path = "data/acquisition_train.csv"
df_raw = pd.read_csv(data_path)
#%%
print(df_raw.head())
print(df_raw.info())
# print missing values ratio
print((df_raw.isnull().sum() / df_raw.shape[0]).sort_values(ascending=False))
# print unique values
print(df_raw.nunique().sort_values(ascending=False))
#%%
# convert target_default to boolean and plot as histogram
print(df_raw.target_default.value_counts())
# get nan count
print(df_raw.target_default.isna().sum())
df_raw.target_default.fillna('False', inplace=True)
df_raw.target_default = df_raw.target_default.replace({'True': True, 'False': False})
print(df_raw.target_default.value_counts())
# plot histogram of df_raw.target_default
plt.figure(figsize=(10, 5))
sns.countplot(x='target_default', data=df_raw)
plt.show()
