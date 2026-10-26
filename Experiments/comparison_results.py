import pandas as pd
from critdd import Diagram
import json
from src.propositional import load_theory
import os
from Data.dataloaders import load_data
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import numpy as np




m_prime={}
all_imbalances=[]
table="&nether&And Or&Or And&NC&MLP+NC&CMR&$F_1(M'(x),M^*(x))$ \\\\ \\hline\n"
all_results=[]
name2ds={"motifs":"motifs","music":"FMA","housing":"ATH", "medical":"derm7pt","faces":"CelebA","objects":"VOC-Pascal","birds":"CUBB","animals":"AWA2", "light":"StarLightCurves"}
no_nn=[0,0]
all_times={"nn":[], "andor":[], "orand":[], "nether":[], "cmr":[], "mnn":[], "full_nether":[]}
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
                    with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_cmr.json") as f:
                        cmr=json.load(f)
                    with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_multimodal.json") as f:
                        multimodal=json.load(f)
                    with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_andor.json") as f:
                        andor=json.load(f)
                    with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_orand.json") as f:
                        orand=json.load(f)
                    with open("Problems/" + name + "/expert_theories/" +str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_expert.json") as f:
                        problem = json.load(f)
                    expert = load_theory(problem["expert_query"])
                    test_data = data.loc[problem["test_indices"]]
                    covered = expert.get_covering_examples(test_data)
                    base_f1=f1_score(problem["test_labels"],[x in covered for x in list(test_data.index)])
                    results.append([nether["f1_score"], andor["f1_score"],orand["f1_score"],nether["nn_f1_score"], multimodal["nn_f1_score"], base_f1, cmr["f1_score"]])




    # subresults={}
    # baseline={}
    # for file in os.listdir("Results/"+name+"/"):
    #     splitted=file.split("_")
    #     if (len(splitted)==11 and splitted[-1]=="new.json") or (len(splitted)==11 and splitted[-1] in ["andor.json", "orand.json"]) or len(splitted)==5:
    #
    #         with open("Results/"+name+"/"+file) as f:
    #             result=json.load(f)
    #
    #         if str(splitted[:4]) not in subresults:
    #
    #             with open("Problems/"+name+"/expert_theories/" + file[:7] + "_expert.json") as f:
    #                 problem = json.load(f)
    #             expert = load_theory(problem["expert_query"])
    #             test_data=data.loc[problem["test_indices"]]
    #             covered=expert.get_covering_examples(test_data)
    #
    #             subresults[str(splitted[:4])]=[0,0,0,0,0,f1_score(problem["test_labels"],[x in covered for x in list(test_data.index)]),0]
    #             all_imbalances.append(sum(problem["test_labels"])/len(problem["test_labels"]))
    #         if len(splitted)==11 and splitted[-1]=="new.json":
    #             subresults[str(splitted[:4])][0]=result["f1_score"]
    #             all_times["nether"].append(result["time"])
    #             all_times["full_nether"].append(sum([sum(result["full_timing"][x]) for x in result["full_timing"]]))
    #             revised=load_theory(result["revised_theory"])
    #             if len(revised.neural_predicates())==0:
    #                 no_nn[0]+=1
    #             no_nn[1]+=1
    #             subresults[str(splitted[:4])][3] = result["nn_f1_score"]
    #             all_times["nn"].append(result["nn_time"])
    #         elif len(splitted)==5 and splitted[-1]=="multimodal.json":
    #             subresults[str(splitted[:4])][4] = result["nn_f1_score"]
    #             all_times["mnn"].append(result["nn_time"])
    #         elif len(splitted)==5 and splitted[-1]=="cmr.json":
    #             subresults[str(splitted[:4])][6] = result["f1_score"]
    #             all_times["cmr"].append(result["time"])
    #         elif splitted[-1]=="andor.json":
    #             subresults[str(splitted[:4])][1] = result["f1_score"]
    #             all_times["andor"].append(result["time"])
    #         elif splitted[-1]=="orand.json":
    #             subresults[str(splitted[:4])][2] = result["f1_score"]
    #             all_times["orand"].append(result["time"])
    #
    #
    # results=[]
    # for i in subresults:
    #     results.append(subresults[i])
    # results=[x for x in results if 0 not in x]
    sharpe=[x[0] for x in results]
    andor=[x[1] for x in results]
    orand=[x[2] for x in results]
    nn=[x[3] for x in results]
    mnn=[x[4] for x in results]
    base=[x[5] for x in results]
    cmr = [x[6] for x in results]

    base=[x[5] for x in results]
    sharpe=[a_i - b_i for a_i, b_i in zip(sharpe, base)]
    andor = [a_i - b_i for a_i, b_i in zip(andor, base)]
    orand = [a_i - b_i for a_i, b_i in zip(orand, base)]
    nn = [a_i - b_i for a_i, b_i in zip(nn, base)]
    mnn = [a_i - b_i for a_i, b_i in zip(mnn, base)]
    cmr = [a_i - b_i for a_i, b_i in zip(cmr, base)]
    print("Sharpe: ", sharpe)

    m_prime[name]=base
    all_results+=results
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
        round(np.std(cmr), 1)) + "\\\\\n")
# print("NO NN:", no_nn)
print(table)
# print(all_imbalances)
ccd=pd.DataFrame(columns=["NeTHeR", "And Or", "Or And", "NN", "MLP+NN", "$M'$","CMR($M'$)"])
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
    "comparison_ccd.tex",
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
# print(len(all_times["cmr"]))
# for x in all_times:
#     print(x, sum(all_times[x])/len(all_times[x])/60)