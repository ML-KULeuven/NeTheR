import pandas as pd
from critdd import Diagram
import json
from src.propositional import load_theory
import os
from matplotlib import pyplot as plt
import numpy as np
all_results=[]
table="& $F_1$ (Freeze) & time (Freeze) $F_1$ (No Freeze) & time (no Freeze) \\\\ \\hline \n"
name2ds={"motifs":"motifs","music":"FMA","housing":"ATH", "medical":"derm7pt","faces":"CelebA","objects":"VOC-Pascal","birds":"CUBB","animals":"AWA2", "light":"StarLightCurves"}
for name in ["housing","animals","faces","birds", "medical","music","motifs","light", "objects"]:
    results=[]
    ### EXPERIMENT 1: freeze, #edits = #removed var
    for width in range(2, 3):
        for depth in range(2, 5):
            for missing_var in range(1, 5):
                for it in range(0, 2):
                    with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(3)+"_"+str(True)+"_"+str(False)+".json") as f:
                        frozen=json.load(f)
                    with open("Results/" + name + "/" + str(depth) + "_" + str(width) + "_" + str(
                                missing_var) + "_" + str(it) + "_" + str(3) + "_" + str(False) + "_" + str(
                                False) + ".json") as f:
                        not_frozen = json.load(f)
                    frozen_query=frozen["revised_theory"]
                    not_frozen_query=not_frozen["revised_theory"]
                    if len([x for x in frozen_query if x[:2] == "NN"]) > 1 and len([x for x in not_frozen_query if x[:2]=="NN"])>1:
                        results.append([frozen["f1_score"],sum(frozen["full_timing"]["training_neural_net"]),not_frozen["f1_score"],sum(not_frozen["full_timing"]["training_neural_net"])])



    freeze_f1=[x[0] for x in results]
    freeze_time=[x[1] for x in results]
    not_freeze_f1=[x[2] for x in results]
    not_freeze_time=[x[3] for x in results]
    all_results.append(results)

    print("######  "+name+"  ######")
    print(len(results))
    table+=name2ds[name]+"&$"+str(round(np.mean(freeze_f1),2))+"\\pm "+str(round(np.std(freeze_f1),2))+"$&$"+str(int(round(np.mean(freeze_time),0)))+ "$&$"+str(round(np.mean(not_freeze_f1),2))+"\\pm "+str(round(np.std(not_freeze_f1),2))+"$&$"+str(int(round(np.mean(not_freeze_time),0)))+"$\\\\\n"

print(table)
