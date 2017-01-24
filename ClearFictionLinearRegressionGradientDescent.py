
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# Import the linear regression class
from sklearn.linear_model import LinearRegression

# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# train test split made easy
from sklearn.cross_validation import train_test_split

#statistics and Integration
from scipy import stats, integrate


# Import preprocessing class
from sklearn import preprocessing



# In[2]:

# read in the cvs
cf_pd = pd.read_csv("cf.csv", verbose=False)

# get the data for predictions
cf_in = pd.read_csv("singleScript.csv", verbose=False)

#create the Profit feature where Profit = 1 if BOX / BUDGET >= 1 and 0 otherwise
cf_pd["Profit"] = cf_pd["BOX"] / cf_pd["BUDGET"]

cf_pd['Profit'].loc[cf_pd['Profit'] >= 1.] = 1
cf_pd['Profit'].loc[cf_pd['Profit'] < 1.] = 0

#80/20 split into train and test
cf_pd_train, cf_pd_test = train_test_split(cf_pd, test_size=0.2)


# In[3]:

#create a dataframe to hold the results
#contains the model name, the features used, movie name, and predicted box
#results = pd.DataFrame(index = ['lr_model', 'rr_model', 'rr_cv_model', 'lasso_cv_model', 'rr_cv_lasso_model'
#                                , 'elastic_net_model', 'elastic_net_model']
#                                , columns=['Features', 'MovieName', 'PredictedBox'])


# In[4]:

my_features = ['BUDGET', 'scrCnt', 'GENRE_DRA', 'GENRE_ROM', 'GENRE_THR', 'GENRE_COM', 'GENRE_HOR', 'GENRE_SCI'
               , 'GENRE_ACT', 'GENRE_FAM', 'NTITLE', 'YEAR', 'avgClrPrem', 'avgImpPrem', 'avgFamSet', 'avgEarExp'
               , 'avgCoAvoid', 'avgIntCon', 'avgSurp', 'avgAntici', 'avgFlhback', 'avgClrMot', 'avgMulDim'
               , 'avgHeroW', 'avgStrNem', 'avgSymHero', 'avgLogic', 'avgCharGrow', 'avgImp', 'avgMulConf'
               , 'avgIntensity', 'avgBuild', 'avgLockIn', 'avgResolut', 'avgBelieve', 'avgSurpEnd', 'NSCENE'
               , 'INTPREC']


# In[5]:

# Normalizing the data
scaler = preprocessing.StandardScaler().fit(cf_pd_train[my_features])
cf_pd_train.loc[:, my_features] = scaler.transform(cf_pd_train[my_features])
cf_pd_test.loc[:, my_features] = scaler.transform(cf_pd_test[my_features])
cf_in.loc[:, my_features] = scaler.transform(cf_in[my_features])


# In[ ]:

from sklearn import linear_model
lr_model = linear_model.LinearRegression()
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
lr_model.fit(cf_pd_train[my_features], cf_pd_train["BOX"])


# The coefficients
print('Coefficients: \n', lr_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((lr_model.predict(cf_pd_test[my_features]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr_model.score(cf_pd_test[my_features], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % lr_model.predict(cf_in[my_features]))

#results.PredictedBox.loc['lr_model'] = lr_model.predict(cf_in[my_features])
#results.MovieName.loc['lr_model'] = cf_in.TITLE.loc[0]
#results.Features.loc['lr_model'] = my_features


# In[ ]:




# In[ ]:

for rr_alpha in [0.1, 1, 10]:
    rr_model = linear_model.Ridge (alpha = rr_alpha)
    rr_model.fit(cf_pd_train[my_features], cf_pd_train["BOX"])

    # The alpha
    print('alpha: ', str(rr_alpha))
    # The coefficients
    print('Coefficients: \n', rr_model.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"
          % np.mean((rr_model.predict(cf_pd_test[my_features]) - cf_pd_test["BOX"]) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % rr_model.score(cf_pd_test[my_features], cf_pd_test["BOX"]))
    # Predict with all features
    print('Predicted Box Office: %.2f' % rr_model.predict(cf_in[my_features]))
    


rr_cv_model = linear_model.RidgeCV()
rr_cv_model.fit(cf_pd_train[my_features], cf_pd_train["BOX"])

# The alpha
print('alpha: ', rr_cv_model.alpha_)
# The coefficients
print('Coefficients: \n', rr_cv_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((rr_cv_model.predict(cf_pd_test[my_features]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rr_cv_model.score(cf_pd_test[my_features], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % rr_cv_model.predict(cf_in[my_features]))


# In[ ]:

lasso_cv_model = linear_model.LassoCV()
lasso_cv_model.fit(cf_pd_train[my_features], cf_pd_train["BOX"])

# The alpha
print('alpha: ', lasso_cv_model.alpha_)
# The coefficients
print('Coefficients: \n', lasso_cv_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((lasso_cv_model.predict(cf_pd_test[my_features]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lasso_cv_model.score(cf_pd_test[my_features], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % lasso_cv_model.predict(cf_in[my_features]))


# In[ ]:

my_features_lasso = ['BUDGET', 'scrCnt', 'GENRE_THR', 'GENRE_COM', 'GENRE_HOR', 'GENRE_FAM', 'NTITLE'
                     , 'avgImpPrem', 'avgEarExp', 'avgLockIn', 'avgResolut', 'NSCENE']

rr_cv_lasso_model = linear_model.RidgeCV()
rr_cv_lasso_model.fit(cf_pd_train[my_features_lasso], cf_pd_train["BOX"])

# The alpha
print('alpha: ', rr_cv_lasso_model.alpha_)
# The coefficients
print('Coefficients: \n', rr_cv_lasso_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((rr_cv_lasso_model.predict(cf_pd_test[my_features_lasso]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rr_cv_lasso_model.score(cf_pd_test[my_features_lasso], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % rr_cv_lasso_model.predict(cf_in[my_features_lasso]))


# In[ ]:

elastic_net_model = linear_model.ElasticNetCV()
elastic_net_model.fit(cf_pd_train[my_features], cf_pd_train["BOX"])

# The alpha
print('alpha: ', elastic_net_model.alpha_)
# The l1 ration
print('l1 ratio: ', elastic_net_model.l1_ratio_)
# The coefficients
print('Coefficients: \n', elastic_net_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((elastic_net_model.predict(cf_pd_test[my_features]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % elastic_net_model.score(cf_pd_test[my_features], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % elastic_net_model.predict(cf_in[my_features]))


# In[ ]:

elastic_net_model = linear_model.ElasticNetCV()
elastic_net_model.fit(cf_pd_train[my_features_lasso], cf_pd_train["BOX"])

# The alpha
print('alpha: ', elastic_net_model.alpha_)
# The l1 ration
print('l1 ratio: ', elastic_net_model.l1_ratio_)
# The coefficients
print('Coefficients: \n', elastic_net_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((elastic_net_model.predict(cf_pd_test[my_features_lasso]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % elastic_net_model.score(cf_pd_test[my_features_lasso], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % elastic_net_model.predict(cf_in[my_features_lasso]))


# In[ ]:

my_features_small = ['BUDGET', 'scrCnt', 'NSCENE']

from sklearn import linear_model
lr_model_small = linear_model.LinearRegression()
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
lr_model_small.fit(cf_pd_train[my_features_small], cf_pd_train["BOX"])


# The coefficients
print('Coefficients: \n', lr_model_small.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((lr_model_small.predict(cf_pd_test[my_features_small]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr_model_small.score(cf_pd_test[my_features_small], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % lr_model_small.predict(cf_in[my_features_small]))


# In[ ]:

lasso_lars_model = linear_model.LassoLarsCV(fit_intercept=True)
lasso_lars_model.fit(cf_pd_train[my_features], cf_pd_train["BOX"])

# The alpha
print('alpha: ', lasso_lars_model.alpha_)
# The coefficients
print('Coefficients: \n', lasso_lars_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((lasso_lars_model.predict(cf_pd_test[my_features]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lasso_lars_model.score(cf_pd_test[my_features], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % lasso_lars_model.predict(cf_in[my_features]))


# In[ ]:

my_lasso_lars_features = ['BUDGET', 'scrCnt', 'GENRE_FAM', 'NTITLE']

from sklearn import linear_model
lr_model_small_lasso_lars = linear_model.LinearRegression()
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
lr_model_small_lasso_lars.fit(cf_pd_train[my_lasso_lars_features], cf_pd_train["BOX"])


# The coefficients
print('Coefficients: \n', lr_model_small_lasso_lars.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((lr_model_small_lasso_lars.predict(cf_pd_test[my_lasso_lars_features]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr_model_small_lasso_lars.score(cf_pd_test[my_lasso_lars_features], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % lr_model_small_lasso_lars.predict(cf_in[my_lasso_lars_features]))


# In[ ]:

from sklearn import linear_model
br_model = linear_model.BayesianRidge()
br_model.fit(cf_pd_train[my_lasso_lars_features], cf_pd_train["BOX"])


# The coefficients
print('Coefficients: \n', br_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((br_model.predict(cf_pd_test[my_lasso_lars_features]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % br_model.score(cf_pd_test[my_lasso_lars_features], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % br_model.predict(cf_in[my_lasso_lars_features]))


# In[ ]:

from sklearn import linear_model
br_model = linear_model.BayesianRidge()
br_model.fit(cf_pd_train[my_features_lasso], cf_pd_train["BOX"])


# The coefficients
print('Coefficients: \n', br_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((br_model.predict(cf_pd_test[my_features_lasso]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % br_model.score(cf_pd_test[my_features_lasso], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % br_model.predict(cf_in[my_features_lasso]))


# In[ ]:

rr_cv_model = linear_model.RidgeCV()
rr_cv_model.fit(cf_pd_train[my_features_lasso], cf_pd_train["BOX"])

# The alpha
print('alpha: ', rr_cv_model.alpha_)
# The coefficients
print('Coefficients: \n', rr_cv_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((rr_cv_model.predict(cf_pd_test[my_features_lasso]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rr_cv_model.score(cf_pd_test[my_features_lasso], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % rr_cv_model.predict(cf_in[my_features_lasso]))


# In[ ]:

rr_cv_model = linear_model.RidgeCV()
rr_cv_model.fit(cf_pd_train[my_lasso_lars_features], cf_pd_train["BOX"])

# The alpha
print('alpha: ', rr_cv_model.alpha_)
# The coefficients
print('Coefficients: \n', rr_cv_model.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((rr_cv_model.predict(cf_pd_test[my_lasso_lars_features]) - cf_pd_test["BOX"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % rr_cv_model.score(cf_pd_test[my_lasso_lars_features], cf_pd_test["BOX"]))
# Predict with all features
print('Predicted Box Office: %.2f' % rr_cv_model.predict(cf_in[my_lasso_lars_features]))


# In[ ]:




# In[ ]:



