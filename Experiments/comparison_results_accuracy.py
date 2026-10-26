import pandas as pd
from critdd import Diagram
import json
from src.propositional import load_theory
import os
from Data.dataloaders import load_data
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from matplotlib import pyplot as plt
import numpy as np

def score(x,y, mode):
    if mode=="precision":
        return precision_score(x,y)
    elif mode=="accuracy":
        return accuracy_score(x,y)
    elif mode=="recall":
        return recall_score(x,y)


mode="recall"

table="&nether&And Or&Or And&NC&Multimodal NN&$M'$ \\\\ \\hline\n"
all_results=[]
name2ds={"housing":"Austin Texas Housing", "medical":"derm7pt","faces":"CelebA","objects":"VOC-Pascal","birds":"CUBB","animals":"AWA2"}

for name in ["housing","animals", "faces","birds", "medical","objects"]:
    data, _,_=load_data(name)
    subresults={}
    baseline={}
    for file in os.listdir("Results/"+name+"/"):
        splitted=file.split("_")
        if (len(splitted)==10 and splitted[-1]=="True.json") or (len(splitted)==11 and splitted[-1] in ["andor.json", "orand.json"]) or len(splitted)==5:

            with open("Results/"+name+"/"+file) as f:
                result=json.load(f)
            with open("Problems/" + name + "/expert_theories/" + file[:7] + "_expert.json") as f:
                problem = json.load(f)
            if str(splitted[:4]) not in subresults:
                expert = load_theory(problem["expert_query"])
                test_data=data.loc[problem["test_indices"]].copy()
                covered=expert.get_covering_examples(test_data)
                subresults[str(splitted[:4])]=[0,0,0,0,0,score(problem["test_labels"],[x in covered for x in list(test_data.index)], mode)]
            if len(splitted)==10  and splitted[-1]=="True.json":
                subresults[str(splitted[:4])][0]=score(problem["test_labels"],result["prediction"], mode)
                subresults[str(splitted[:4])][3] = score(problem["test_labels"],result["nn_prediction"], mode)
            elif len(splitted)==5:
                subresults[str(splitted[:4])][4] = score(problem["test_labels"],result["nn_prediction"], mode)
            elif splitted[-1]=="andor.json":
                subresults[str(splitted[:4])][1] = score(problem["test_labels"],result["prediction"], mode)
            elif splitted[-1]=="orand.json":
                subresults[str(splitted[:4])][2] = score(problem["test_labels"],result["prediction"], mode)


    results=[]
    for i in subresults:
        results.append(subresults[i])
    results=[x for x in results if 0 not in x]
    sharpe=[x[0] for x in results]
    andor=[x[1] for x in results]
    orand=[x[2] for x in results]
    nn=[x[3] for x in results]
    mnn=[x[4] for x in results]
    base=[x[5] for x in results]

    all_results+=results
    print("######  " + name + "  ######")
    print("Sharpe: ", round(np.mean(sharpe), 2), "+-", round(np.std(sharpe), 2))
    print("And OR: ", round(np.mean(andor), 2), "+-", round(np.std(andor), 2))
    print("Or And: ", round(np.mean(orand), 2), "+-", round(np.std(orand), 2))
    print("NP: ", round(np.mean(nn), 2), "+-", round(np.std(nn), 2))
    print("Multimodal: ", round(np.mean(mnn), 2), "+-", round(np.std(mnn), 2))
    print("Input: ", round(np.mean(base), 2), "+-", round(np.std(base), 2))
    table += (name2ds[name] + "&" + str(round(np.mean(sharpe), 2)) + "$\pm$" + str(round(np.std(sharpe), 2)) + "&" + str(round(np.mean(andor), 2)) + "$\pm$" + str(round(np.std(andor), 2)) + "&"+ str(round(np.mean(orand), 2)) + "$\pm$" + str(round(np.std(orand), 2)) + "&"+ str(round(np.mean(nn), 2)) + "$\pm$" + str(round(np.std(nn), 2))+ "&"+ str(round(np.mean(mnn), 2)) + "$\pm$" + str(round(np.std(mnn), 2))+ "&"+ str(round(np.mean(base), 2)) + "$\pm$" + str(round(np.std(base), 2)) + "\\\\\n")
print(table)
ccd=pd.DataFrame(columns=["nether", "And Or", "Or And", "NC", "Multimodal", "$M'$"])
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
    "comparison_"+mode+"_ccd.tex",
    alpha = .05,
    adjustment = "holm",
    reverse_x = True,
    # axis_options = {"title": "critdd"},
)
#     print(len(sharpe))
#     print(sharpe)
#     print("nether:", sum(sharpe)/len(sharpe))
#     print("And or:",sum(andor)/len(andor))
#     print("Or and:",sum(orand)/len(sharpe))
#     print("NN: ", sum(nn)/len(nn))
#     print("Multimodal nn: ", sum(mnn)/len(mnn))
#     print("Input: ", sum(base)/len(base))
#
# plt.scatter(list(range(0,len(sharpe))), list(sharpe), label="nether (Ours)")
# plt.scatter(list(range(0,len(andor))), list(andor), label="And or")
# plt.scatter(list(range(0,len(orand))), list(orand), label="Or and")
# plt.scatter(list(range(0,len(nn))), list(nn), label="NN")
# plt.scatter(list(range(0,len(mnn))), list(mnn), label="Multimodal NN")
# plt.scatter(list(range(0,len(base))), list(base), label="Input")
#
# plt.xlabel("Instantie ID")
# plt.ylabel("f1-score")
# plt.ylim([0,1.1])
# plt.legend()
# plt.savefig("plots/music_comparison.png")
# with open("Problems/animals/expert_theories/3_2_2_0_expert.json") as f:
#     problem=json.load(f)
# print(load_theory(problem["original_query"]).show())
# print("---------")
# print(load_theory(problem["expert_query"]).show())