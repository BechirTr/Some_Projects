from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas 
from scipy import interpolate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer,ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def Count_(df,col):
  ss={}
  for g in df[col]:
    if g in ss:
      ss[g]+=1
    else:
      ss[g]=1
  return ss
  
def Get_Genders(X_df):
  #X_df[(X_df.Gender != "Male") and (X_df.Gender != "Female") and (X_df.Gender != 'Non-binary/third gender') and (X_df.Gender != 'Not Asked')]["Gender"] ='Not Asked'
  
  res = X_df
  res['Gender'] = res['Gender'].fillna('Unknown')
  res['Gender'] = res["Gender"].astype('category').cat.codes
  
  return res


def HMC_transformer(X_df):
  res = X_df.replace({'HowManyCompanies':{'Not Asked' : 0 ,
                      "1 (this is the only company where I've had this kind of position)":1,
                      '2 (I worked at another similar position elsewhere before this one)':2,
                      '6 or more':6}})
  res['HowManyCompanies'] = res['HowManyCompanies'].fillna(0)
  res['HowManyCompanies'] = res["HowManyCompanies"].astype('category').cat.codes
  return res
def OPOYT_transformer(X_df):
  res = X_df.replace({"OtherPeopleOnYourTeam":{'None': 0 ,'More than 5':6} })
  res['OtherPeopleOnYourTeam'] = res['OtherPeopleOnYourTeam'].fillna(0)
  res['OtherPeopleOnYourTeam'] = res["OtherPeopleOnYourTeam"].astype('category').cat.codes
  return res
def Country_transformer(X_df):
  res = X_df
  res['Country'] = X_df['Country'].fillna('Unknown')
  res['Country'] = res["Country"].astype('category').cat.codes
  return res
def transformer_1(df,threshold=150):
  '''
  se = Count_(df,"OtherDatabases")
  most_used_db = []
  for i,j in se.items():
    if j>threshold:
      if type(i)==str:
        most_used_db.append(i)
  new_cols={}
  for i in most_used_db:
    new_cols[i]=np.zeros(df.shape[0])

  for i,line in enumerate(df['OtherDatabases']):
    if type(line) is str :
      databases = line.split(', ')
      for db in databases:
        if db in new_cols:
          new_cols[db][i]=1

  for i in most_used_db:
    df[i] = new_cols[i]
'''
  df['OtherDatabases'] = df['OtherDatabases'].fillna('')
  df['OtherDatabases'] = df['OtherDatabases'].str.split(', ').str.len()
  df['PrimaryDatabase'] = df['PrimaryDatabase'].fillna('Unknown')
  df["PrimaryDatabase"] = df["PrimaryDatabase"].astype('category')
  df["PrimaryDatabase"] = df["PrimaryDatabase"].cat.codes
  return df

  
transformer_K = FunctionTransformer(
    lambda X_df: transformer_1(X_df)
)
transformer_KS = make_column_transformer((transformer_K,['OtherDatabases','PrimaryDatabase']))
def encode_feature(df,feature):
  df[feature] = df[feature].fillna("Unkown")
  ord_enc = OrdinalEncoder()
  return ord_enc.fit_transform(df[[feature]]).reshape((len(df[[feature]]),1))

def standard_scale(df,feature):
  res = df[[feature]].to_numpy().reshape((len(df[[feature]]),1))
  return (res - np.mean(res))/np.std(res)

cols = ["Education",
        "HoursWorkedPerWeek",
        'Certifications',
        'EducationIsComputerRelated']

transformer_HMC = FunctionTransformer(
    lambda X_df: HMC_transformer(X_df)
)
transformer_OPOYT = FunctionTransformer(
    lambda X_df: OPOYT_transformer(X_df)
)
transformer_country = FunctionTransformer(
    lambda X_df: Country_transformer(X_df)
)
transformer_b = make_column_transformer((transformer_HMC,['HowManyCompanies']),(transformer_OPOYT,['OtherPeopleOnYourTeam']),(transformer_country,['Country']))

transformer_Education = FunctionTransformer(lambda df: encode_feature(df,'Education'))
transformer_Certifications = FunctionTransformer(lambda df: encode_feature(df,'Certifications'))
transformer_EducationIsComputerRelated = FunctionTransformer(lambda df: encode_feature(df,'EducationIsComputerRelated'))
transformer_HoursWorkedPerWeek = FunctionTransformer(lambda df: standard_scale(df,'HoursWorkedPerWeek'))

transformer_MK = make_column_transformer(
    (transformer_Education,['Education']),
    (transformer_HoursWorkedPerWeek,['HoursWorkedPerWeek']),
    (transformer_Certifications,['Certifications']),
    (transformer_EducationIsComputerRelated,['EducationIsComputerRelated']), 
    ('passthrough', cols)
)

sector_dict={'Education (K-12, college, university)': 0,
 'Federal government': 1,
 'Local government': 2,
 'Non-profit': 3,
 'Private business': 4,
 'State/province government': 5,
 'Student': 0}
 ##############

def JobTitle(df,job_dict):
  jt = Counter(df.JobTitle)
  job_dict = {}
  for job, count in jt.items():
    if count < 20 :
      job_dict[job] = "Other"
    else :
      job_dict[job] = job

  jobs = df.JobTitle.apply(lambda x:job_dict[x]).to_numpy()
  return (jobs.reshape((len(jobs),1)))

def sector(df,sector_dict):
  sectors =  df.EmploymentSector.apply(lambda x:sector_dict[x]).to_numpy()
  return (sectors.reshape((len(sectors),1)))



transformer_JobTitle = FunctionTransformer(lambda df: JobTitle(df,job_dict))
transformer_sector = FunctionTransformer(lambda df: sector(df,sector_dict))

# categorical columns with ordinal encoding
cat_cols = ["ManageStaff", "EmploymentStatus"]
cat_pipeline = make_pipeline(
    SimpleImputer(strategy='constant'),
    OrdinalEncoder(),
)

transformer_nk = make_column_transformer(
    (cat_pipeline, cat_cols),
    (transformer_JobTitle,["JobTitle"]),
    (transformer_sector,["EmploymentSector"])
)

def get_elements(key): 
  L = key.split(",")
  Result = []
  i = 0 
  while i < len(L):
    word = L[i]
    if word.find("(") != -1: 
      while word.find(")") == -1 : 
        i += 1 
        word = word +","+ L[i]
    i += 1
    Result.append(word) 
  return Result

def get_elts_count(X, col): 
  elts_counter = Counter(X[col])
  elts_pairs = {}
  for key, value in elts_counter.items():
    if type(key) != str:
      continue
    elts = get_elements(key)
    for elt in elts:
      if elt not in elts_pairs.keys(): 
        elts_pairs[elt] = value
      else:
        elts_pairs[elt] += value 
  return elts_pairs

def get_dummy_jobs(X_df, num): # Here X_df is composed of a single column
  feature_array = np.zeros((len(X_df.index), num))
  col_name = X_df.columns[0]
  elts_pairs  = get_elts_count(X_df, col_name)
  elts_pairs = [(key,value) for key, value  in elts_pairs.items()]
  elts_pairs =  sorted(elts_pairs, key=lambda tup: tup[1], reverse= True)
  elts_kept = [pair[0]  for pair in elts_pairs[:num] ]
  X_df = X_df[col_name]
  indexes = list(X_df.index.values)
 
  for p in range(len(X_df)):
    i = indexes[p]
    if (type(X_df[i]) != str):
      for  j in range(len(elts_kept)):
        feature_array[p][j] = np.nan
      continue
    elts = get_elements(X_df[i])
    for  j in range(len(elts_kept)): 
      if elts_kept[j] in elts : 
        feature_array[p][j] = 1 
  return feature_array
transformer_dummy_jobs = FunctionTransformer(
    lambda X_df: get_dummy_jobs(X_df, 1)
)
def get_top_jobs(X_df, num): # Here X_df is composed of a single column
  feature_array = np.zeros((len(X_df), num))
  col_name = X_df.columns[0]
  elts_pairs = Counter(X_df[col_name])
  elts_pairs = [(key,value) for key, value  in elts_pairs.items()]
  elts_pairs =  sorted(elts_pairs, key=lambda tup: tup[1], reverse= True)
  elts_kept = [pair[0]  for pair in elts_pairs[:num - 1] ]
  if 'Other' in elts_kept: 
    elts_kept.append(elts_pairs[num][0])
  else : 
    elts_kept.append('Other')
  Other_index = elts_kept.index('Other')
  X_df = X_df[col_name]
  indexes = list(X_df.index.values)
 
  for p in range(len(X_df)):
    i = indexes[p]
    if (type(X_df[i]) != str):
      for  j in range(len(elts_kept)):
        feature_array[p][j] = np.nan
      continue
    elt = X_df[i]
    if  elt in elts_kept:
      for j in range(num): 
        if elt == elts_kept[j]: 
          feature_array[p][j] = 1
          break
    else: 
      feature_array[p][num-1] = 1
  return feature_array

transformer_JobTitle = FunctionTransformer(
    lambda X_df: get_top_jobs(X_df, 4)
)
transformer_Gender = FunctionTransformer(
    lambda X_df: Get_Genders(X_df)
)

job_transformer = Pipeline(steps=[
    ('transform', transformer_dummy_jobs),
    ('imputer', SimpleImputer(strategy= 'most_frequent'))])


jobTitle_transfomer = Pipeline(steps=[
    ('transform', transformer_JobTitle),
    ('imputer', SimpleImputer(strategy= 'most_frequent'))])
carreer_transformer = Pipeline(steps=[
    ('one_hot_encoder', OneHotEncoder()),
    ('imputer', SimpleImputer(strategy= 'most_frequent')),          
])
LAJ_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder()),
    ('imputer', SimpleImputer(strategy= 'most_frequent')),          
])
othercols = ['Survey Year',
 'YearsWithThisDatabase',
 'YearsWithThisTypeOfJob',
 'DatabaseServers']
 
preprocessor = ColumnTransformer(
    transformers=[
        ('0', job_transformer, ['OtherJobDuties']),
        ('1', job_transformer, ['KindsOfTasksPerformed']),
        ('2', jobTitle_transfomer, ['JobTitle']),
        ('3', carreer_transformer, ['CareerPlansThisYear']),
        ('4', transformer_HMC, ['HowManyCompanies']),
        ('5', transformer_OPOYT, ['OtherPeopleOnYourTeam']),
        ('6', transformer_country, ['Country']),
        ('7',transformer_Education,['Education']),
        ('8',transformer_HoursWorkedPerWeek,['HoursWorkedPerWeek']),
        ('9',transformer_Certifications,['Certifications']),
        ('10',transformer_EducationIsComputerRelated,['EducationIsComputerRelated']),
        ('11',cat_pipeline, cat_cols),
        ('12',transformer_sector,["EmploymentSector"]),
        ('13',transformer_K,['OtherDatabases','PrimaryDatabase']),
        ('14',LAJ_transformer,['LookingForAnotherJob']),
        ('15',transformer_Gender,["Gender"]),
        ('16','passthrough',othercols)

    ])


pipe = make_pipeline(
    
    preprocessor,
    
    RandomForestRegressor()
)
def get_estimator():
    return pipe

