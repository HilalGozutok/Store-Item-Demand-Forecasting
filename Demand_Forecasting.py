#####################################################
# Store Item Demand Forecasting
#####################################################

#####################################################
# Libraries
#####################################################

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
from helpers.eda import *
from helpers.data_prep import *

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


#####################################################
# Loading the data
#####################################################

train = pd.read_csv('Store Item Demand Forecasting/dataset/train.csv', parse_dates=['date'])
test = pd.read_csv('Store Item Demand Forecasting/dataset/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('Store Item Demand Forecasting/dataset/sample_submission.csv')
df = pd.concat([train, test], sort=False)

#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max()

check_df(train)

check_df(test)

check_df(sample_sub)

check_outlier(df, "sales")

missing_values_table(df)


# Satış dağılımı nasıl?
df[["sales"]].describe().T

# Kaç store var?
df[["store"]].nunique()

# Kaç item var?
df[["item"]].nunique()

# Her store'da eşit sayıda mı eşsiz item var?
df.groupby(["store"])["item"].nunique()

# Peki her store'da eşit sayıda mı sales var?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# mağaza-item kırılımında satış istatistikleri
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

#####################################################
# FEATURE ENGINEERING
#####################################################

#####################################################
# Date Features
#####################################################

df.head()
df.shape

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek + 1
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)
df.head(20)


df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


#####################################################
# Random Noise
#####################################################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


#####################################################
# Lag/Shifted Features
#####################################################


df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

check_df(df)
df["sales"].head(10)
df["sales"].shift(1).values[0:10]


pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})



df.groupby(["store", "item"])['sales'].head()

df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))
def lag_features(dataframe, lags):
    dataframe = dataframe.copy()
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

df.head()

df[df["sales"].isnull()]

df[df["date"] == "2017-10-02"]

# pd.to_datetime("2018-01-01") - pd.DateOffset(91)

#####################################################
# Rolling Mean Features
#####################################################

# Hareketli Ortalamalar

df["sales"].head(10)
df["sales"].rolling(window=2).mean().values[0:10]
df["sales"].rolling(window=3).mean().values[0:10]
df["sales"].rolling(window=5).mean().values[0:10]


pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})

def roll_mean_features(dataframe, windows):
    dataframe = dataframe.copy()
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546])

df.head()

#####################################################
# Exponentially Weighted Mean Features
#####################################################


pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm01": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})



def ewm_features(dataframe, alphas, lags):
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales']. \
                    transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

check_df(df)
df.columns


#####################################################
# One-Hot Encoding
#####################################################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])


#####################################################
# Converting sales to log(1+sales)
#####################################################

df['sales'] = np.log1p(df["sales"].values)

#####################################################
# Custom Cost Function
#####################################################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


#####################################################
# MODEL VALIDATION
#####################################################

#####################################################
# Time-Based Validation Sets
#####################################################

# Kaggle test seti tahmin edilecek değerler: 2018'in ilk 3 ayı.
test["date"].min(), test["date"].max()
train["date"].min(), train["date"].max()

# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]
train["date"].min(), train["date"].max()

# 2017'nin ilk 3'ayı validasyon seti.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

df.columns

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


#####################################################
# LightGBM Model
#####################################################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)


y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
smape(np.expm1(y_pred_val), np.expm1(Y_val))



##########################################
# Değişken önem düzeyleri
##########################################

def plot_lgb_importances(model, plot=False, num=10):
    from matplotlib import pyplot as plt
    import seaborn as sns
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)

plot_lgb_importances(model, plot=True, num=30)



lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()


##########################################
# Final Model
##########################################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}


# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

# Create submission
submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)
submission_df.to_csv('kaggle_submission.csv', index=False)
