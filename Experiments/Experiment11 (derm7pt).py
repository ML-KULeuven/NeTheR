from pysdd.sdd import Vtree, SddManager, SddNode
import pandas as pd


from src.construct_model import prop2sdd
from src.neural_architectures.convolutional import res_cnn, multimodal_cnn
from src.propositional import VAR, OR, AND
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from Data.dataloaders import load_data
from src.DT import learn_decision_tree
from sklearn.model_selection import train_test_split
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader
import weka.core.jvm as jvm
import arff
from tqdm import tqdm
from src.propositional import clean_theory
from sklearn import tree
from sklearn.tree import export_text
import os
from src.Main import approximate_revision
from src.train_model import validate_model, train_model
import torch
from CMR.cmr_src import CMR,netherEncoder
import json
device=[3,]


train_indices=list(pd.read_csv("Data/medical/train_indexes.csv")["indexes"])
validation_indices=list(pd.read_csv("Data/medical/valid_indexes.csv")["indexes"])
test_indices=list(pd.read_csv("Data/medical/test_indexes.csv")["indexes"])
training_indices=train_indices+validation_indices
data,cat_cols,drop_cols=load_data("medical_diagnosis")


new_train_data=data.iloc[train_indices]
new_validation_data=data.iloc[validation_indices]
new_test_data=data.iloc[test_indices]
training_data=data.iloc[training_indices]


# jvm.stop()
# jvm.start(system_cp=True, packages=True)
all_scores=[]

all_f1=[]
all_acc=[]
all_acc_0=[]
train_labels=[int(x==2) for x in list(new_train_data["diagnosis"])]
training_labels=[int(x==2) for x in list(training_data["diagnosis"])]
validation_labels=[int(x==2) for x in list(new_validation_data["diagnosis"])]
test_labels = [int(x == 2) for x in list(new_test_data["diagnosis"])]
for i in range(50,51):#50
    for x in range(25,26):#2
        for z in range(10,11):#7
            dt, clf=learn_decision_tree(training_data.copy(), training_labels, cat_cols, i,x,z)
            train_data=new_train_data.copy().drop(columns=["diagnosis"])
            validation_data = new_validation_data.copy().drop(columns=["diagnosis"])
            test_data=new_test_data.copy().drop(columns=["diagnosis"])
            pred = clf.predict(test_data[cat_cols])
            train_pred = clf.predict(train_data[cat_cols])
            valid_pred=clf.predict(validation_data[cat_cols])
            test_pred=clf.predict(test_data[cat_cols])
                    # dt.set_covering_examples(validation_data[cat_cols])
            print(i,x,z,f1_score(test_labels, test_pred),len(dt.nodes()))
print("TREE NAME: ", dt.name())

if True:
    # PARAMETERS
    edit_distance = 10
    k = 5  # alpha estimation
    # alpha=.8
    freeze = True  # previous changes are "locked"

    mode = "medical_diagnosis"
    epochs = 25
    already_done = False


    print("@@@@@@@@@@@@@@@@@@@@ NEW PROBLEM: ", 1)


    input_data = (train_data, validation_data)
    input_labels = (train_labels, validation_labels)
    conditions = [x.condition for x in dt.variables()]


    best_result,_ = approximate_revision(input_data, dt,input_labels,cat_cols, edit_distance, k, mode,freeze, epochs=epochs, device=device, symb_allowed=True)

    print("BEST RESULT: ", best_result)
    if best_result[1] == None:  # only removals
        print("No added nets")
        covered_test_indices = best_result[2].get_covering_examples(test_data)
        prediction = [x in covered_test_indices for x in test_data.index]
        valid_prediciton=[x in covered_test_indices for x in test_data.index]
        train_prediction = [x in covered_test_indices for x in test_data.index]
    else:
        print("Prediction on test data")
        prediction = validate_model(best_result[1], test_data)
        valid_prediction=validate_model(best_result[1], validation_data)
        train_prediction = validate_model(best_result[1], train_data)
    print("---- WRITING RESULTS ----")
    result = {}
    result["dt_f1_score"]=f1_score(test_labels, pred)
    result["f1_score"] = f1_score(test_labels, prediction)
    result["train_dt_f1_score"] = f1_score(train_labels, train_pred)
    result["train_f1_score"]=f1_score(train_labels, train_prediction)
    result["validation_dt_f1_score"] = f1_score(validation_labels, valid_pred)
    result["validation_f1_score"] = f1_score(validation_labels, valid_prediction)
    result["dt_precision_score"]=precision_score(test_labels, pred)
    result["dt_recall_score"]=recall_score(test_labels, pred)
    result["dt_prediction"]=[int(x) for x in pred]

    result["precision_score"] = precision_score(test_labels, prediction)
    result["recall_score"] = recall_score(test_labels, prediction)
    result["original_theory"] = dt.save()
    result["revised_theory"] = best_result[2].save()
    result["prediction"] = prediction
    print(result)
    with open("Results/medical_binary/full_concepts.json", "w") as f:
        json.dump(result, f)
