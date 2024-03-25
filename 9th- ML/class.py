
import numpy as np 	#Array		
import matplotlib.pyplot as plt		
import pandas as pd		
#--------------------------------------------

# import the dataset & divided my dataset into independe & dependent

dataset = pd.read_csv(r"C:\Users\kdata\Desktop\KODI WORK\1. NARESH\1. MORNING BATCH\N_Batch -- 10.00AM\3. Mar\8th- ML\5. Data preprocessing\Data.csv")

X = dataset.iloc[:, :-1].values	

y = dataset.iloc[:,3].values  

#--------------------------------------------

from sklearn.impute import SimpleImputer # SPYDER 4 

imputer = SimpleImputer() 

imputer = imputer.fit(X[:,1:3]) 

X[:, 1:3] = imputer.transform(X[:,1:3])
