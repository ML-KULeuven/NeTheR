import os
import json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from Data.dataloaders import load_data


def score(x,y):
    return roc_auc_score(x,y)

test_indices=list(pd.read_csv("nether/Data/medical/test_indexes.csv")["indexes"])


data,cat_cols,drop_cols=load_data("medical_diagnosis")



nether=[]
nn=[]
multimodal=[]
dt=[]
zero=[]

with open("Results/medical_binary/full_concepts.json") as f:
    res=json.load(f)
with open("Results/medical_binary/cmr.json") as f:
    cmr=json.load(f)

new_test_data = data.iloc[test_indices]
pred=res["prediction"]
dt_pred=res["dt_prediction"]
cmr_pred=cmr["prediction"]
test_labels = [int(x == 2) for x in list(new_test_data["diagnosis"])]

print("Nether: ", score(test_labels,pred))
print("DT: ", score(test_labels,dt_pred))
print("CMR: ", score(test_labels, cmr_pred))

