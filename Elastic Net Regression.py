# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 07:14:14 2023

@author: M GNANESHWARI
"""
# grid search hyperparameters for the elastic net
from numpy import arange
from pandas import read_csv
import sklearn
from sklearn.model_selection import GridSearchCV
What is GridSearchCV used for?
What is GridSearchCV used for? GridSearchCV is a technique for finding the optimal parameter values from a given set of parameters in a grid. It's essentially a cross-validation technique.
from sklearn.model_selection import RepeatedKFold
Repeated K-Fold cross validator.

Repeats K-Fold n times with different randomization in each repetition.
  Repeated k-fold cross-validation provides a way to improve the estimated performance of a machine learning model. This involves simply repeating the cross-validation procedure multiple times and reporting the mean result across all folds from all runs.
from sklearn.linear_model import ElasticNet
#What is an elastic net regression?
#What is elastic net regression? Elastic net regression is a linear regression technique that uses a penalty term to shrink the coefficients of the predictors. The penalty term is a combination of the l1-norm (absolute value) and the l2-norm (square) of the coefficients, weighted by a parameter called alpha.
# load the dataset
# =============================================================================
# import os  
# path = os.path.abspath(r'file path')
# f = open(path)
# print(f)
# =============================================================================
What is an elastic net regression?
What is elastic net regression? Elastic net regression is a linear regression technique that uses a penalty term to shrink the coefficients of the predictors. The penalty term is a combination of the l1-norm (absolute value) and the l2-norm (square) of the coefficients, weighted by a parameter called alpha.
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = pd.read_csv(url, header=None)
data = dataframe.values
print(data)
X, y = data[:, :-1], data[:, -1]
# define model
model = ElasticNet()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
grid['l1_ratio'] = arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
Running the example will evaluate each combination of configurations using repeated cross-validation.

You might see some warnings that can be safely ignored, such as:

Objective did not converge. You might want to increase the number of iterations.
Your specific results may vary given the stochastic nature of the learning algorithm. Try running the example a few times.

In this case, we can see that we achieved slightly better results than the default 3.378 vs. 3.682. Ignore the sign; the library makes the MAE negative for optimization purposes.

We can see that the model assigned an alpha weight of 0.01 to the penalty and focuses exclusively on the L2 penalty.

MAE: -3.378
Config: {'alpha': 0.01, 'l1_ratio': 0.97}
The scikit-learn library also provides a built-in version of the algorithm that automatically finds good hyperparameters via the ElasticNetCV class.

To use this class, it is first fit on the dataset, then used to make a prediction. It will automatically find appropriate hyperparameters.

By default, the model will test 100 alpha values and use a default ratio. We can specify our own lists of values to test via the “l1_ratio” and “alphas” arguments, as we did with the manual grid search.

The example below demonstrates this.

# use automatically configured elastic net algorithm
from numpy import arange
from pandas import read_csv
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
import urllib.request
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
#dataframe = urllib.request.Request(url)
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
ratios = arange(0, 1, 0.01)
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
# fit model
model.fit(X, y)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)
print('l1_ratio_: %f' % model.l1_ratio_)
