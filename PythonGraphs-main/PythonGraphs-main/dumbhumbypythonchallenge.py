# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 16:44:57 2021

@author: M GNANESHWARI
"""

import pandas as pd
Exclusion_df = pd.read_csv("Desktop\Exclusion.csv",
                         keep_default_na=False, na_values=[""])
print(surveys_df)
Products_df = pd.read_csv("Desktop\Products.csv",
                         keep_default_na=False, na_values=[""])
print(Products_df)
Relevency_table_df = pd.read_csv("Desktop\Relevency_table.csv",
                         keep_default_na=False, na_values=[""])
print(Relevency_table_df)
df= Relevency_table_df.groupby('customers').apply(count(Relevency_table_df['customers'])>=3 and count(Relevency_table_df['customers'])<=8)
