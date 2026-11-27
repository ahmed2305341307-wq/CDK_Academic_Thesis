import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from .config import *
def load_data():
    df=pd.read_csv(ROOT/'data/raw/kidney_disease.csv'); y=df.classification.str.strip().map({'ckd':1,'notckd':0})
    if y.isna().any(): raise ValueError('Unknown label')
    X=df[NUM+CAT].copy()
    for c in NUM: X[c]=pd.to_numeric(X[c],errors='coerce')
    for c in CAT:
        valid=['normal','abnormal'] if c in ['rbc','pc'] else ['present','notpresent'] if c in ['pcc','ba'] else ['good','poor'] if c=='appet' else ['yes','no']
        s=X[c].astype('string').str.strip().str.lower(); X[c]=s.where(s.isin(valid)).astype(object); X[c]=X[c].where(X[c].notna(),np.nan)
    if X.duplicated().any(): raise ValueError('Duplicate full predictor rows require grouped split')
    return X,y.astype(int)
def preprocess(cols):
    return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),[c for c in cols if c in NUM]),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),[c for c in cols if c in CAT])])
