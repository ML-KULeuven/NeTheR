import pandas as pd
from critdd import Diagram
import json
from src.propositional import load_theory
import os
from Data.dataloaders import load_data
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import numpy as np
all_results=[]
dt=0
cmr=0
counter=0
table="&NeTheR&And Or&Or And&NC&MLP+NC&CMR \\\\ \\hline\n"
all_results=[]
name2ds={"motifs":"motifs","music":"FMA","housing":"ATH", "medical":"derm7pt","faces":"CelebA","objects":"VOC-Pascal","birds":"CUBB","animals":"AWA2", "light":"StarLightCurves"}

for name in ["housing","animals", "faces","birds", "medical","music","motifs","light", "objects"]:
    subresults = {}
    data, _, _ = load_data(name)
    for file in [x for x in os.listdir("Results/"+name+"/") if "dt.json" in x]:
        dt_file="Results/"+name+"/"+file
        cmr_file="Results/"+name+"/"+file[:7]+"_dt_cmr.json"
        andor_files = [x for x in os.listdir("Results/" + name + "/") if "dt_andor" in x and file[:7] in x]
        orand_files = [x for x in os.listdir("Results/" + name + "/") if "dt_orand" in x and file[:7] in x]

        multimodal_file="Results/"+name+"/"+file[:7]+"_multimodal.json"
        nn_file=dt_file.replace("_dt.json","_3_True_True.json")

        if False not in [os.path.isfile(x) for x in [dt_file, cmr_file, multimodal_file,nn_file]] and len(andor_files)==1 and len(orand_files)==1:
            print(dt_file)
            counter+=1
            with open(dt_file) as f:
                dt=json.load(f)
            with open(cmr_file) as f:
                cmr=json.load(f)
            with open("Results/"+name+"/"+andor_files[0]) as f:
                andor=json.load(f)
            with open("Results/"+name+"/"+orand_files[0]) as f:
                orand = json.load(f)
            with open(multimodal_file) as f:
                multimodal = json.load(f)
            with open(nn_file) as f:
                nn = json.load(f)




            splitted=dt_file[8:].split("_")
            with open("Problems/" + name + "/expert_theories/" + file[:7] + "_expert.json") as f:
                problem = json.load(f)
            expert = load_theory(dt["decision_tree"])
            test_data = data.loc[problem["test_indices"]]
            covered = expert.get_covering_examples(test_data)
            f1_dt = f1_score(problem["test_labels"], [x in covered for x in list(test_data.index)])
            if f1_dt<0.6:
                print("WE HAVE ONE: ", f1_dt)

            subresults[str(splitted[:4])] = [dt["f1_score"], andor["f1_score"], orand["f1_score"], nn["nn_f1_score"], multimodal["nn_f1_score"], f1_dt, cmr["f1_score"]]
        else:
            print("ERROR: ", [os.path.isfile(x) for x in [dt_file, cmr_file, multimodal_file,nn_file]], andor_files, orand_files)
    results = []
    for i in subresults:
        results.append(subresults[i])
    # results=[x for x in results if 0 not in x]
    sharpe = [x[0] for x in results]
    andor = [x[1] for x in results]
    orand = [x[2] for x in results]
    nn = [x[3] for x in results]
    mnn = [x[4] for x in results]
    base = [x[5] for x in results]
    cmr = [x[6] for x in results]

    base = [x[5] for x in results]
    sharpe = [a_i - b_i for a_i, b_i in zip(sharpe, base)]
    andor = [a_i - b_i for a_i, b_i in zip(andor, base)]
    orand = [a_i - b_i for a_i, b_i in zip(orand, base)]
    nn = [a_i - b_i for a_i, b_i in zip(nn, base)]
    mnn = [a_i - b_i for a_i, b_i in zip(mnn, base)]
    cmr = [a_i - b_i for a_i, b_i in zip(cmr, base)]
    print("Sharpe: ", sharpe)

    all_results += results
    print("######  " + name + "  ######")
    # print("Sharpe: ", round(np.mean(sharpe), 2), "+-", round(np.std(sharpe), 2))
    # print("And OR: ", round(np.mean(andor), 2), "+-", round(np.std(andor), 2))
    # print("Or And: ", round(np.mean(orand), 2), "+-", round(np.std(orand), 2))
    # print("NP: ", round(np.mean(nn), 2), "+-", round(np.std(nn), 2))
    # print("Multimodal: ", round(np.mean(mnn), 2), "+-", round(np.std(mnn), 2))
    # print("Input: ", round(np.mean(base), 2), "+-", round(np.std(base), 2))
    table += (name2ds[name] + "&\\bf" + str(round(np.mean(sharpe), 2)) + "$\\pm$" + str(
        round(np.std(sharpe), 1)) + "&" + str(round(np.mean(andor), 2)) + "$\\pm$" + str(
        round(np.std(andor), 1)) + "&" + str(round(np.mean(orand), 2)) + "$\\pm$" + str(
        round(np.std(orand), 1)) + "&" + str(round(np.mean(nn), 2)) + "$\\pm$" + str(
        round(np.std(nn), 1)) + "&" + str(round(np.mean(mnn), 2)) + "$\\pm$" + str(
        round(np.std(mnn), 1)) + "&" + str(round(np.mean(cmr), 2)) + "$\\pm$" + str(
        round(np.std(cmr), 1))  + "\\\\\n")

    print("&" + str(round(np.mean(base), 2)) + "$\\pm$" + str(
        round(np.std(base), 1)))

ccd = pd.DataFrame(columns=["NeTheR", "And Or", "Or And", "NN", "MLP+NN", "$M'$", "CMR($M'$)"])
for i in range(0, len(all_results)):
    ccd.loc[len(ccd)] = all_results[i]
diagram = Diagram(
    ccd.to_numpy(),
    treatment_names=ccd.columns,
    maximize_outcome=True
)

# inspect average ranks and groups of statistically indistinguishable treatments
diagram.average_ranks  # the average rank of each treatment
diagram.get_groups(alpha=.05, adjustment="holm")

# export the diagram to a file
diagram.to_file(
    "comparison_ccd_dt.tex",
    alpha=.05,
    adjustment="holm",
    reverse_x=True,
    # axis_options = {"title": "critdd"},
)

print(table)
print("Succeeded: ", counter, "/", str(177))