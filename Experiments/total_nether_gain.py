import pandas as pd
from critdd import Diagram
import json
from src.propositional import load_theory
import os
from Data.dataloaders import load_data
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import numpy as np

full_f1=[]


for name in ["housing","animals", "faces","birds", "medical","music","motifs","light", "objects"]:
    data, _,_=load_data(name)
    results = []
    ### EXPERIMENT 1: freeze, #edits = #removed var
    for width in range(2, 3):
        for depth in range(2, 5):
            for missing_var in range(1, 5):
                for it in range(0, 2):
                    with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(3)+"_"+str(True)+"_"+str(True)+".json") as f:
                        nether=json.load(f)
                    with open("Problems/" + name + "/expert_theories/" +str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_expert.json") as f:
                        problem = json.load(f)
                    expert = load_theory(problem["expert_query"])
                    test_data = data.loc[problem["test_indices"]]
                    covered = expert.get_covering_examples(test_data)
                    base_f1=f1_score(problem["test_labels"],[x in covered for x in list(test_data.index)])
                    full_f1.append(nether["f1_score"]-base_f1)

for name in ["housing", "animals", "faces", "birds", "medical", "music", "motifs", "light", "objects"]:
    subresults = {}
    data, _, _ = load_data(name)
    for file in [x for x in os.listdir("Results/" + name + "/") if "dt.json" in x]:
        dt_file = "Results/" + name + "/" + file



        with open(dt_file) as f:
            dt = json.load(f)

        splitted = dt_file[8:].split("_")
        with open("Problems/" + name + "/expert_theories/" + file[:7] + "_expert.json") as f:
            problem = json.load(f)
        expert = load_theory(dt["decision_tree"])
        test_data = data.loc[problem["test_indices"]]
        covered = expert.get_covering_examples(test_data)
        f1_dt = f1_score(problem["test_labels"], [x in covered for x in list(test_data.index)])
        full_f1.append(dt["f1_score"]-f1_dt)
print(sum(full_f1)/len(full_f1))