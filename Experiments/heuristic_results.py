import pandas as pd
from critdd import Diagram
import json
from src.propositional import load_theory
import os
from Data.dataloaders import load_small_animal_data
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import numpy as np
all_results=[]
name2ds={"motifs":"motifs","music":"FMA","housing":"ATH", "medical":"derm7pt","faces":"CelebA","objects":"VOC-Pascal","birds":"CUBB","animals":"AWA2", "light": "StarLightCurves"}

table="&Sharpe & Sortino&Worst case $(\\alpha=0)$&Expected value $(\\alpha=0.5)$&Best case $(\\alpha=1)$ \\\\ \\hline\n"
for name in [ "housing","animals","faces","birds", "medical","music","motifs","light", "objects"]:
    results=[]
    ### EXPERIMENT 1: freeze, #edits = #removed var
    for width in range(2, 3):
        for depth in range(2, 5):
            for missing_var in range(1, 5):
                for it in range(0, 2):
                    try:
                        with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_max.json") as f:
                            max_result=json.load(f)
                        with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_mean.json") as f:
                            mean_result=json.load(f)
                        with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_min.json") as f:
                            min_result=json.load(f)
                        with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(3)+"_"+str(True)+"_"+str(True)+".json") as f:
                            nether_result=json.load(f)
                        with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(3)+"_"+str(True)+"_"+str(True)+"_sortino.json") as f:
                            sortino_result=json.load(f)
                        results.append([nether_result["f1_score"],sortino_result["f1_score"], min_result["f1_score"], mean_result["f1_score"], max_result["f1_score"]])
                    except Exception as e:
                        print("missing: ", e)



    all_results+=results
    sharpe=[x[0] for x in results]
    sortino = [x[1] for x in results]
    min_res=[x[2] for x in results]
    mean_res=[x[3] for x in results]
    max_res=[x[4] for x in results]
    print("######  " + name + "  ######")
    print("Sharpe: ", round(np.mean(sharpe), 2), "+-", round(np.std(sharpe), 2))
    print("Sortino: ", round(np.mean(sortino), 2), "+-", round(np.std(sortino), 2))
    print("Min: ", round(np.mean(min_res), 2), "+-", round(np.std(min_res), 2))
    print("Mean: ", round(np.mean(mean_res), 2), "+-", round(np.std(mean_res), 2))
    print("Max: ", round(np.mean(max_res), 2), "+-", round(np.std(max_res), 2))
    table += (name2ds[name] + "&" + str(round(np.mean(sharpe), 2)) + "$\pm$" + str(round(np.std(sharpe), 2)) + "&" + str(round(np.mean(sortino), 2)) + "$\pm$" + str(round(np.std(sortino), 2))+ "&" + str(round(np.mean(min_res), 2)) + "$\pm$" + str(round(np.std(min_res), 2)) + "&"
                                                                                                                                                                                                   + str(round(np.mean(mean_res), 2)) + "$\pm$" + str(round(np.std(mean_res), 2)) + "&"
                                                                                                                                                                                                                                                                                      + str(round(np.mean(max_res), 2)) + "$\pm$" + str(round(np.std(max_res), 2)) + "\\\\\n")

ccd=pd.DataFrame(columns=["Sharpe","Sortino", "Worst case", "Expected value", "Best case"])
for i in range(0,len(all_results)):
    ccd.loc[len(ccd)] = all_results[i]
diagram = Diagram(
    ccd.to_numpy(),
    treatment_names = ccd.columns,
    maximize_outcome = True
)

# inspect average ranks and groups of statistically indistinguishable treatments
diagram.average_ranks # the average rank of each treatment
diagram.get_groups(alpha=.05, adjustment="holm")

# export the diagram to a file
diagram.to_file(
    "heuristic_ccd_sortino.tex",
    alpha = .05,
    adjustment = "holm",
    reverse_x = True,
)
print(table)
