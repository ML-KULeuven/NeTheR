from threnpi.Data.dataloaders import load_data
from threnpi.src.train_model import train_model, validate_model
from threnpi.src.neural_architectures.convolutional import res_cnn
from threnpi.src.neural_architectures.mlp import mlp
from sklearn.metrics import f1_score, precision_score, recall_score
from threnpi.src.Main import approximate_revision
from threnpi.src.propositional import load_theory
import json
import time
import os


name="animals"
#Full data is the dataframe with all categorical + image dat
#cat_cols the categorical column names
#drop_cols the columns that could be dropped during problem generation
full_data, cat_cols, drop_cols=load_data(name)

#parameters to select which problem instance
width=2
depth=2
missing_var=1
it=0

with open("Problems/"+name+"/expert_theories/" + str(depth) + "_" + str(width) + "_" + str(
        missing_var) + "_" + str(it) + "_expert.json") as f:
    problem = json.load(f)
    expert_theory = load_theory(problem["expert_query"])
    original_theory = load_theory(problem["original_query"])

# PARAMETERS
edit_distance = len(original_theory.variables()) - len(expert_theory.variables())  # how many edits are ideally needed
k = 3  # alpha estimation
h = 1  # no hill climbing
p = 5  # patience
merge = False  # does nothing right now
freeze = True  # previous changes are "locked"
mode="animal_conv"
epochs = 50

#loading all the data
training_indices=problem["train_indices"]
training_labels=problem["train_labels"]
training_data=full_data.loc[training_indices].copy()
validation_indices = problem["validation_indices"]
validation_labels = problem["validation_labels"]
validation_data = full_data.loc[validation_indices].copy()
test_indices = problem["test_indices"]
test_labels = problem["test_labels"]
test_data = full_data.loc[test_indices].copy()

input_data=(training_data, validation_data)
input_labels=(training_labels, validation_labels)

#ThReNPI
best_result=approximate_revision(input_data, expert_theory,input_labels, edit_distance, k,h,p,merge, mode,freeze, epochs=epochs)
