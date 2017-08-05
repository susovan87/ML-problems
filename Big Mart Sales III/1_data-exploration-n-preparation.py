#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# =============================================================================
# 1. Hypothesis must be ready
# =============================================================================

# # Don't forget to reset environment
# # IPython console command - '%reset'
# from IPython import get_ipython
# get_ipython().magic('reset -sf')

#np.set_printoptions(threshold=np.nan)




# =============================================================================
# 2. Data Exploration
# =============================================================================
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# combine train and test sets into one to perform feature engineering
train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
print(train.shape, test.shape, data.shape)


# check missing values
data.apply(lambda x: sum(x.isnull()))


# To get a overview
data.head(n=10)
data.describe().transpose()


# To find nominal (categorical) variable, check number of unique values
data.apply(lambda x: len(x.unique()))


# Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
# Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
# Print frequency of categories
for col in categorical_columns:
    print('\nFrequency of Categories: %s'%col)
    print(data[col].value_counts())




# =============================================================================
# 3. Data Cleaning
# =============================================================================
# Determine the average weight per item:
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')


# Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull() 


#Impute data and check #missing values before and after imputation to confirm
print('Orignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])
print('Final #missing: %d'%sum(data['Item_Weight'].isnull()))


from scipy.stats import mode

#Determing the mode for each
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x.dropna()).mode[0]) )
print('Mode for each Outlet_Type:')
print(outlet_size_mode)


#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Outlet_Size'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print('\nOrignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool,'Outlet_Size'] = data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print( sum(data['Outlet_Size'].isnull()) )




# =============================================================================
# 4. Feature Engineering
# =============================================================================
# before combining variables, validate
# checking mean sales by type of store, as significant difference, leaving as it is
data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')



# modify Item_Visibility as min 0 makes no practical sense, treating as missing value
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
miss_bool = (data['Item_Visibility'] == 0)
print('Number of 0 values initially: %d'%sum(miss_bool))
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
print('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))



# determine another variable with visibility means ratio
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']], axis=1)
print(data['Item_Visibility_MeanRatio'].describe())



## Create a broad category of Type of Item
# get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
# rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()



# Determine the years of operation of a store
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()



## Modify categories of Item_Fat_Content
#Change categories of low fat:
print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())

print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())

#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()



## Numerical and One-Hot Coding of Categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size',
           'Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])

data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type',
                                     'Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
data.dtypes
data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)


## Exporting Data
#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)
