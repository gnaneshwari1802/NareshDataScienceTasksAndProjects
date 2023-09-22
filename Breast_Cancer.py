# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:38:08 2023

@author: M GNANESHWARI
"""
Import Libraries
import  numpy  as  np
import  pandas  as  pd
import  matplotlib.pyplot  as  plt
import  seaborn  as  sns
%matplotlib  inline
import  warnings
warnings.filterwarnings('ignore')
#display  image  using  python
from  IPython.display  import  Image
url='https://d2jx2rerrg6sh3.cloudfront.net/image-handler/ts/20170215120249/ri/673/picture/2017/2/shutterstock_576066646.jpg'
Image(url,height=300,width=400)
Import Dataset
dataset=pd.read_csv("https://raw.githubusercontent.com/gnaneshwari1802/28th/main/projects/KNN/brest%20cancer.txt?token=GHSAT0AAAAAACGCKUCF5BPT2434GOPF2BZOZINGTRQ") 
dataset
#Exploratory Data Analysis
#Now, I will explore the the to gain insight about the data.
#  view  dimensions  of  dataset
dataset.shape
"""
We can see that there are 699 instances and 11 attributes in the dataset.
in the dataset description, it is given that there are 10 attributes and 1class which is the target varia attribiutes and 1target variable.
"""
#View  top  5rows  of  this  dataset
dataset.head()
"""
Rename column names
We can see that the dataset doesn't have proper column names.The columns are mainly labelled a should give proper names to the columns.
"""
col_names  = ['Id',  'Clump_thickness',  'Uniformity_Cell_Size',  'Uniformity_Cell_Shape',  'Margi', 'Single_Epithelial_Cell_Size',  'Bare_Nuclei',  'Bland_Chromatin',  'Normal_Nucleol']

dataset.columns  = col_names 
dataset.columns

#We can see that column names are renamed. Now the columns have meaningful names.
#  let's  agian  preview  the  dataset
dataset.head()
Drop redundant columns
We should drop any redundant columns from the dataset which doesn't have any predictive power. column.
#  drop  Id  column  from  dataset
dataset.drop('Id',  axis=1,  inplace=True)
View summary of this dataset
#  view  summary  of  dataset
dataset.info()
"""We can see that the Id column has been removed from this dataset.
Frequency distribution of values in variables"""
for  var  in  dataset.columns: 
    print(dataset[var].value_counts())
"""    
The distribution of values shows that data type of Bare_Nuclei is of type integer but summary of the is type object.    
Convert data type of Bare_Nuclei to integer
"""
dataset['Bare_Nuclei']  = pd.to_numeric(dataset['Bare_Nuclei'],  errors='coerce')
"""Check data types of colmuns of the dataframe"""
dataset.dtypes
"""
Now, we can see that all the columns of the dataframe are numeric type.
Explore problems within variables
Now, i will explore problems within variables.
Missing values in variables
"""
#  check  missing  values  in  variables
dataset.isnull().sum()
We can see that the Bare_Nuclei cloumn contains missing values. we need to dig deeper to find the values.
#  check  `na`  values  in  the  dataframe
dataset.isna().sum()
We can see that the Bare_Nuclei column contains 16 nan values.
#  check  frequency  distribution  of  `Bare_Nuclei`  column
dataset['Bare_Nuclei'].value_counts()

#  check  unique  values  in  `Bare_Nuclei`  column
dataset['Bare_Nuclei'].unique()

We can see that there are nan values in the Bare_Nuclei column.
#  check  for  nan  values  in  `Bare_Nuclei`  column
dataset['Bare_Nuclei'].isna().sum()
#  check  unique  values  in  `Bare_Nuclei`  column
dataset['Bare_Nuclei'].unique()
We can see that there are nan values in the Bare_Nuclei column.
#  check  for  nan  values  in  `Bare_Nuclei`  column
dataset['Bare_Nuclei'].isna().sum()
We can see that there are 16 nan vaules in the dataset. I will impute missing values after dividing th and testset.
Check frequency distribution of target variable class
#  view  frequency  distribution  of  values  in  `Class`  variable
dataset['Class'].value_counts()
Check percentage of frequency ditribution of class
#  view  percentage  of  frequency  distribution  of  values  in  `Class`  variable
dataset['Class'].value_counts()/np.float(len(dataset))
Outliers in numerical variables
#  view  summary  statistics  in  numerical  variables
print(round(dataset.describe(),2))
Knn algorithm is roboost outliers.
Univariate plots
Check the distribution of variables
Now, i will plot histogram to check variable distributions to find out if they are normal or skewed.
#  plot  histograms  of  the  variables
plt.rcParams['figure.figsize']=(30,25)
dataset.plot(kind='hist',  bins=10,  subplots=True,  layout=(5,2),  sharex=False,  sharey=False) 
plt.show()
"""
We can see that all the variables in the dataset are positively skewed.
Multivariate plots
Estimating correlation coefficients
Our dataset is very small. So we can compute the standard correlation coefficient between every pa compute it using the dataset.corr() method
"""
correlation=dataset.corr()
Our target is class. So we should check how each attribute coorelates with the class variable.
correlation['Class'].sort_values(ascending=False)
Discover patterns and relationships
An important step in EDA is to discover patterns and relationships between variables in the dataset. heatmap to explore the partterns and relationships in the dataset.
Correlation Heat Map
plt.figure(figsize=(10,8))
plt.title('Correlation  of  Attributes  with  Class  variable')
a  = sns.heatmap(correlation,  square=True,  annot=True,  fmt='.2f',  linecolor='white') a.set_xticklabels(a.get_xticklabels(),  rotation=90) a.set_yticklabels(a.get_yticklabels(),  rotation=30)
plt.show()
Declare feature vector and target variable
X  = dataset.drop(['Class'],  axis=1) 
y  = dataset['Class']
Split data into separate training and test set
#  split  X  and  y  into  training  and  testing  sets
from  sklearn.model_selection  import  train_test_split
X_train,  X_test,  y_train,  y_test  = train_test_split(X,  y,  test_size  = 0.2,  random_state  = 0)
#  check  the  shape  of  X_train  and  X_test
X_train.shape,  X_test.shape
Feature Engineering
Feature Engineering** is the process of transforming raw data into useful features that help us to un better and increase its predictive power. I will carry out feature engineering on different types of vari
#  check  data  types  in  X_train
X_train.dtypes
Engineering missing values in variables
#  check  missing  values  in  numerical  variables  in  X_train
X_train.isnull().sum()
#  check  missing  values  in  numerical  variables  in  X_test
X_test.isnull().sum()
#  print  percentage  of  missing  values  in  the  numerical  variables  in  training  set
for  col  in  X_train.columns:
if  X_train[col].isnull().mean()>0:
print(col,  round(X_train[col].isnull().mean(),4))
Assumption
I assume that the data are missing completely at random (MCAR). There are two methods which ca missing values. One is mean or median imputation and other one is random sample imputation. Wh the dataset, we should use median imputation. So, I will use median imputation because median im outliers.
I will impute missing values with the appropriate statistical measures of the data, in this case media done over the training set, and then propagated to the test set. It means that the statistical measure missing values both in train and test set, should be extracted from the train set only. This is to avoid
#  impute  missing  values  in  X_train  and  X_test  with  respective  column  median  in  X_train
for  df1  in  [X_train,  X_test]: for  col  in  X_train.columns:
col_median=X_train[col].median()
df1[col].fillna(col_median,  inplace=True)
#  check  again  missing  values  in  numerical  variables  in  X_train
X_train.isnull().sum()
#  check  missing  values  in  numerical  variables  in  X_test
X_test.isnull().sum()
We can see that there are no missing values in X_train and X_test.
X_train.head()
X_test.head()
We now have training and testing set ready for model building. Before that, we should map all the fe same scale. It is called 'feature scaling'.
Feature Scaling
cols  = X_train.columns
from  sklearn.preprocessing  import  StandardScaler scaler  = StandardScaler()
X_train  = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
X_train  = pd.DataFrame(X_train,  columns=[cols])
X_test  = pd.DataFrame(X_test,  columns=[cols])
X_train.head()
"""

We now have 'X_train' dataset ready to be fed into the Logistic Regression classifier.
Fit K Neighbours Classifier to the training eet
"""
#  import  KNeighbors  ClaSSifier  from  sklearn
from  sklearn.neighbors  import  KNeighborsClassifier
#  instantiate  the  model
knn  = KNeighborsClassifier(n_neighbors=3) #  fit  the  model  to  the  training  set knn.fit(X_train,  y_train)
KNeighborsClassifier

KNeighborsClassifier(n_neigh bors=3)
Predict test-set results

y_pred  = knn.predict(X_test)
y_pred
Predict_proba method
predict_proba method gives the probabilities for the target variable(2 and 4) in this case, in array f
#  probability  of  getting  output
knn.predict_proba(X_test)[:,0]	as  2  -  benign  cancer

#  probability  of  getting  output  as  4  -  malignant  cancer
knn.predict_proba(X_test)[:,1]
Check accuracy score
from  sklearn.metrics  import  accuracy_score
print('Model  accuracy  score:  {0:0.4f}'.  format(accuracy_score(y_test,  y_pred)))
Compare the train-set and test-set accuracy Now, I will compare the train-set and test-set accuracy
y_pred_train  = knn.predict(X_train)
print('Training-set  accuracy  score:  {0:0.4f}'.  format(accuracy_score(y_train,  y_pred_train)))
Check for overfitting and underfitting
#  print  the  scores  on  training  and  test  set
print('Training  set  score:  {:.4f}'.format(knn.score(X_train,  y_train))) print('Test  set  score:  {:.4f}'.format(knn.score(X_test,  y_test)))
The training-set accuracy score is 0.9803 while the test-set accuracy to be 0.9714. These two value So, there is no question of overfitting.
Compare model accuracy with null accuracy
So, the model accuracy is 0.9714. But, we cannot say that our model is very good based on the ab compare it with the null accuracy. Null accuracy is the accuracy that could be achieved by always frequent class.
So, we should first check the class distribution in the test set
#  check  class  distribution  in  test  set
y_test.value_counts()
We can see that the occurences of most frequent class is 85. So, we can calculate null accuracy by number of occurences.
#  check  null  accuracy  score
null_accuracy  = (85/(85+55))
print('Null  accuracy  score:  {0:0.4f}'.  format(null_accuracy))
We can see that our model accuracy score is 0.9714 but null accuracy score is 0.6071. So, we can Nearest Neighbors model is doing a very good job in predicting the class labels.
Rebuild kNN Classification model using k=5
#  instantiate  the  model  with  k=5
knn_5  = KNeighborsClassifier(n_neighbors=5) #  fit  the  model  to  the  training  set knn_5.fit(X_train,  y_train)
#  predict  on  the  test-set
y_pred_5  = knn_5.predict(X_test)
print('Model  accuracy  score  with  k=5  :  {0:0.4f}'.  format(accuracy_score(y_test,  y_pred_5)))
Rebuild kNN Classification model using k=6
#  instantiate  the  model  with  k=6
knn_6  = KNeighborsClassifier(n_neighbors=6) 
#  fit  the  model  to  the  training  set knn_6.fit(X_train,  y_train)
#  predict  on  the  test-set
y_pred_6  = knn_6.predict(X_test)
print('Model  accuracy  score  with  k=6  :  {0:0.4f}'.  format(accuracy_score(y_test,  y_pred_6)))
Rebuild kNN Classification model using k=7
#  instantiate  the  model  with  k=7
knn_7  = KNeighborsClassifier(n_neighbors=7) #  fit  the  model  to  the  training  set knn_7.fit(X_train,  y_train)
#  predict  on  the  test-set
y_pred_7  = knn_7.predict(X_test)
print('Model  accuracy  score  with  k=7  :  {0:0.4f}'.  format(accuracy_score(y_test,  y_pred_7)))
Rebuild kNN Classification model using k=8
#  instantiate  the  model  with  k=8
knn_8  = KNeighborsClassifier(n_neighbors=8) #  fit  the  model  to  the  training  set knn_8.fit(X_train,  y_train)
#  predict  on  the  test-set
y_pred_8  = knn_8.predict(X_test)
print('Model  accuracy  score  with  k=8  :  {0:0.4f}'.  format(accuracy_score(y_test,  y_pred_8)))
Rebuild kNN Classification model using k=9
#  instantiate  the  model  with  k=9
knn_9  = KNeighborsClassifier(n_neighbors=9) #  fit  the  model  to  the  training  set knn_9.fit(X_train,  y_train)
#  predict  on  the  test-set
y_pred_9  = knn_9.predict(X_test)
print('Model  accuracy  score  with  k=9  :  {0:0.4f}'.  format(accuracy_score(y_test,  y_pred_9)))

Interpretation
Our original model accuracy score with k=3 is 0.9714. Now, we can see that we get same accuracy k=5. But, if we increase the value of k further, this would result in enhanced accuracy.
With k=6,7,8 we get accuracy score of 0.9786. So, it results in performance improvement. If we increase k to 9, then accuracy decreases again to 0.9714.
Confusion matrix
A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusi clear picture of classification model performance and the types of errors produced by the model. It g correct and incorrect predictions broken down by each category. The summary is represented in a t
Four types of outcomes are possible while evaluating a classification model performance. These fo described below:-
True Positives (TP) – True Positives occur when we predict an observation belongs to a certain cla actually belongs to that class.
True Negatives (TN) – True Negatives occur when we predict an observation does not belong to a observation actually does not belong to that class.
False Positives (FP) – False Positives occur when we predict an observation belongs to a certain actually does not belong to that class. This type of error is called Type I error.

False Negatives (FN) – False Negatives occur when we predict an observation does not belong to observation actually belongs to that class. This is a very serious error and it is called Type II error.

#  Print  the  Confusion  Matrix  with  k  =3  and  slice

#  Print  the  Confusion  Matrix  with  k  =3  and  slice
from  sklearn.metrics  import  confusion_matrix 
cm  = confusion_matrix(y_test,  y_pred) 
print('Confusion  matrix\n\n',  cm) 
print('\nTrue  Positives(TP)  =  ',  
      cm[0,0]) print('\nTrue  Negatives(TN)  =  ',  
                     cm[1,1]) print('\nFalse  Positives(FP)  =  ',  
                                    cm[0,1]) 
                                    print('\nFalse  Negatives(FN)  =  ',  cm[1,0])	
it	
into	
four	
pieces

Confusion  matrix	
cm  = confusion_matrix(y_test,  y_pred) print('Confusion  matrix\n\n',  cm) print('\nTrue  Positives(TP)  =  ',  cm[0,0]) print('\nTrue  Negatives(TN)  =  ',  cm[1,1]) print('\nFalse  Positives(FP)  =  ',  cm[0,1]) print('\nFalse  Negatives(FN)  =  ',  cm[1,0])
The confusion matrix shows 83  +  53  =  136  correct  predictions  and 2  +  2  =  4  incorrect  predicti
In this case, we have
'True Positives' (Actual Positive:1 and Predict Positive:1) - 83 'True Negatives' (Actual Negative:0 and Predict Negative:0) - 53
'False Positives' (Actual Negative:0 but Predict Positive:1) - 2 '(Type I error)' 'False Negatives' (Actual Positive:1 but Predict Negative:0) - 2 '(Type II error)'
#  Print  the  Confusion  Matrix  with  k  =7  and  slice  it  into  four  pieces
cm_7  = confusion_matrix(y_test,  y_pred_7) print('Confusion  matrix\n\n',  cm_7) print('\nTrue  Positives(TP)  =  ',  cm_7[0,0]) print('\nTrue  Negatives(TN)  =  ',  cm_7[1,1]) print('\nFalse  Positives(FP)  =  ',  cm_7[0,1]) print('\nFalse  Negatives(FN)  =  ',  cm_7[1,0])
The above confusion matrix shows '83 + 54 = 137 correct predictions' and '2 + 1 = 4 incorrect predi In this case, we have
'True Positives' (Actual Positive:1 and Predict Positive:1) - 83 'True Negatives' (Actual Negative:0 and Predict Negative:0) - 54
'False Positives' (Actual Negative:0 but Predict Positive:1) - 2 '(Type I error)' 'False Negatives' (Actual Positive:1 but Predict Negative:0) - 1 '(Type II error)'
#  visualize  confusion  matrix  with  seaborn  heatmap
plt.figure(figsize=(6,4))
cm_matrix  = pd.DataFrame(data=cm_7,  columns=['Actual  Positive:1',  'Actual  Negative:0'],
index=['Predict  Positive:1',  'Predict  Negative:0']) sns.heatmap(cm_matrix,  annot=True,  fmt='d',  cmap='YlGnBu')

Classification Report
Classification report is another way to evaluate the classification model performance. It displays t and support scores for the model. I have described these terms in later.
from  sklearn.metrics  import  classification_report print(classification_report(y_test,  y_pred_7))
Classification accuracy
TP  = cm_7[0,0] TN  = cm_7[1,1] FP  = cm_7[0,1] FN  = cm_7[1,0]
#  print  classification  accuracy
classification_accuracy  = (TP  + TN)  / float(TP  + TN  + FP  + FN) print('Classification  accuracy  :  {0:0.4f}'.format(classification_accuracy))
Classification error
#  print  classification  error
classification_error  = (FP  + FN)  / float(TP  + TN  + FP  + FN) print('Classification  error  :  {0:0.4f}'.format(classification_error))

Classification  error  :  0.0429
Precision
Precision can be defined as the percentage of correctly predicted positive outcomes out of all the p outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (T
So, Precision identifies the proportion of correctly predicted positive outcome. It is more concerned than the negative class.
Mathematically, 'precision' can be defined as the ratio of 'TP to (TP + FP)'.
#  print  precision  score
precision  = TP  / float(TP  + FP) print('Precision  :  {0:0.4f}'.format(precision))
Recall
recall  = TP  / float(TP  + FN)

print('Recall  or  Sensitivity  :  {0:0.4f}'.format(recall))
True Positive Rate
true_positive_rate  = TP  / float(TP  + FN)
print('True  Positive  Rate  :  {0:0.4f}'.format(true_positive_rate))
False Positive Rate
false_positive_rate  = FP  / float(FP  + TN)
print('False  Positive  Rate  :  {0:0.4f}'.format(false_positive_rate))
Specificity
specificity  = TN  / (TN  + FP)
print('Specificity  :  {0:0.4f}'.format(specificity))
Adjusting the classification threshold level
#  print  the  first  10  predicted  probabilities  of  two  classes-  2  and  4
y_pred_prob  = knn.predict_proba(X_test)[0:10] y_pred_prob
#  store  the  probabilities  in  dataframe
y_pred_prob_df  = pd.DataFrame(data=y_pred_prob,  columns=['Prob  of  -  benign  cancer  (2)',  'Prob y_pred_prob_df
                                                        #  print  the  first  10  predicted  probabilities  for  class  4  -  Probability  of  malignant  cancer
knn.predict_proba(X_test)[0:10,  1]

#  store  the  predicted  probabilities  for  class  4  -  Probability  of  malignant  cancer
y_pred_1  = knn.predict_proba(X_test)[:,  1]   
#  plot  histogram  of  predicted  probabilities #  adjust  figure  size plt.figure(figsize=(6,4))
#  adjust  the  font  size
plt.rcParams['font.size']  = 12 #  plot  histogram  with  10  bins plt.hist(y_pred_1,  bins  = 10)
#  set  the  title  of  predicted  probabilities
plt.title('Histogram  of  predicted  probabilities  of  malignant  cancer')
#  set  the  x-axis  limit
plt.xlim(0,1)
#  set  the  title
plt.xlabel('Predicted  probabilities  of  malignant  cancer') plt.ylabel('Frequency')
ROC-AUC
#  plot  ROC  Curve
from  sklearn.metrics  import  roc_curve
fpr,  tpr,  thresholds  = roc_curve(y_test,  y_pred_1,  pos_label=4) plt.figure(figsize=(6,4))
plt.plot(fpr,  tpr,  linewidth=2) plt.plot([0,1],  [0,1],  'k--'  )
plt.rcParams['font.size']  = 12
plt.title('ROC  curve  for  Breast  Cancer  kNN  classifier') plt.xlabel('False  Positive  Rate  (1  -  Specificity)') plt.ylabel('True  Positive  Rate  (Sensitivity)') plt.show()
ROC AUC
ROC AUC stands for Receiver Operating Characteristic - Area Under Curve. It is a technique to performance. In this technique, we measure the area under the curve (AUC) . A perfect classifier wil to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.
#  compute  ROC  AUC
from  sklearn.metrics  import  roc_auc_score ROC_AUC = roc_auc_score(y_test, y_pred_1) print('ROC  AUC  :  {:.4f}'.format(ROC_AUC))
Interpretation
ROC AUC is a single number summary of classifier performance. The higher the value, the bet
ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a whether it is benign or malignant cancer.
#  calculate  cross-validated  ROC  AUC
from  sklearn.model_selection  import  cross_val_score
Cross_validated_ROC_AUC  = cross_val_score(knn_7,  X_train,  y_train,  cv=5,  scoring='roc_auc').m print('Cross  validated  ROC  AUC  :  {:.4f}'.format(Cross_validated_ROC_AUC))
Cross  validated  ROC  AUC  :  
    