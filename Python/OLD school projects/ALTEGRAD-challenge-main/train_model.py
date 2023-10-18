# basic libraries import
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline

# import the model
from model import Model_reg 

################### Training phase ###################
# read training data
print("load data")
df_train = pd.read_csv('data/train.csv', dtype={'authorID': np.int64, 'h_index': np.float32})
n_train = df_train.shape[0]

# read Node2Vec
df = pd.read_csv("data/graph_embedding.csv")
df = df.T 
df = df.reset_index()
df = df.rename({"index":"authorID"},axis=1)
df["authorID"] = df['authorID'].astype("int64")

# read embeddings of abstracts   
embeddings = pd.read_csv("data/author_embedding.csv", header=None)
embeddings = embeddings.rename(columns={0: "authorID"})

# read garph features 
ftr_train = pd.read_csv("data/graph_features.csv", header= None)
ftr_train = ftr_train 
ftr_train = ftr_train.reset_index()
ftr_train = ftr_train.rename(columns={0: "authorID"})
# remove extra features
ftr_train = ftr_train.drop(columns= ["index",1]) 
ftr_train["authorID"] = ftr_train['authorID'].astype("int64")

# merge the dataframes
df_train = df_train.merge(embeddings, on= "authorID")
df_train = df_train.merge(df, on= "authorID")
df_train = df_train.merge(ftr_train, on = "authorID")
print("data has been merged")


# create the X_train and y_train matrix
X_train = df_train.iloc[:,2:].values
y_train = df_train.iloc[:,1].values

# Define and train the model
reg =  Pipeline(steps=[('scale', StandardScaler()),
                      ('model', Model_reg())])
print("model training begin")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_train)
mae = mean_absolute_error(y_train, y_pred)

print("MAE on train:", mae)
print(reg.get_params()["model"].abc)
################### Prediction phase ###################  

# read test data
print("load test data")
df_test = pd.read_csv('data/test.csv', dtype={'authorID': np.int64})
n_test = df_test.shape[0]

# merge the dataframes
df_test = df_test.merge(embeddings, on= "authorID")
df_test = df_test.merge(df, on= "authorID")
df_test = df_test.merge(ftr_train, on = "authorID")
print("test set merged")

# create the X_test matrix
X_test = df_test.iloc[:,2:].values

# predict the h-index
print("Predicting")
y_pred = reg.predict(X_test)

# write the CSV file
df_test['h_index_pred'].update(pd.Series(np.round_(y_pred, decimals=3)))
df_test.loc[:,["authorID","h_index_pred"]].to_csv('test_predictions.csv', index=False)
