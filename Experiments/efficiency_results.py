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
all_results=[]
name2ds={"motifs":"motifs","music":"FMA","housing":"ATH", "medical":"derm7pt","faces":"CelebA","objects":"VOC-Pascal","birds":"CUBB","animals":"AWA2", "light":"StarLightCurves"}
nether=[[] for x in range(0,10)]
CMR=[[] for x in range(0,10)]
NN=[[] for x in range(0,10)]
multimodal=[[] for x in range(0,10)]
exps=["motifs","housing","animals", "faces","birds", "medical","music","light", "objects"]
for name in exps:
    data, _, _ = load_data(name)
    for width in range(2, 3):
        for depth in range(2, 5):
            for missing_var in range(1, 5):
                for it in range(0, 2):
                    with open("Problems/" + name + "/expert_theories/" + str(depth) + "_" + str(width) + "_" + str(
                                        missing_var) + "_" + str(it) + "_expert.json") as f:
                        problem = json.load(f)
                        expert = load_theory(problem["expert_query"])
                        test_data = data.loc[problem["test_indices"]]
                        covered = expert.get_covering_examples(test_data)
                        base=0#f1_score(problem["test_labels"],[x in covered for x in list(test_data.index)])
                    for percentage in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

                        with open("Results/" + name + "/" + str(depth) + "_" + str(width) + "_" + str(
                                        missing_var) + "_" + str(it) + "_training_"+str(percentage)+".json", "r") as f:
                            training_result=json.load(f)
                        print("Results/" + name + "/" + str(depth) + "_" + str(width) + "_" + str(
                                        missing_var) + "_" + str(it) + "_training_"+str(percentage)+".json")
                        nether[int(10*percentage)-1].append(training_result["f1_score"]-base)
                        NN[int(10*percentage) - 1].append(training_result["nn_f1_score"]-base)
                        multimodal[int(10*percentage) - 1].append(training_result["multi_f1_score"]-base)
                        CMR[int(10*percentage) - 1].append(training_result["cmr_f1_score"]-base)
    for file in [x for x in list(os.listdir("Results/" + name + "/")) if x.endswith("True.json")]:
        with open("Problems/" + name + "/expert_theories/" +file[:7] + "_expert.json") as f:
            problem = json.load(f)
            expert = load_theory(problem["expert_query"])
            test_data = data.loc[problem["test_indices"]]
            covered = expert.get_covering_examples(test_data)
            base = 0#f1_score(problem["test_labels"], [x in covered for x in list(test_data.index)])
        with open("Results/" + name + "/"+file,"r") as f:
            res=json.load(f)
        nether[9].append(res["f1_score"]-base)
        NN[9].append(res["nn_f1_score"]-base)
    for file in [x for x in list(os.listdir("Results/" + name + "/")) if x.endswith("multimodal.json")]:
        with open("Problems/" + name + "/expert_theories/" + file[:7] + "_expert.json") as f:
            problem = json.load(f)
            expert = load_theory(problem["expert_query"])
            test_data = data.loc[problem["test_indices"]]
            covered = expert.get_covering_examples(test_data)
            base = 0#f1_score(problem["test_labels"], [x in covered for x in list(test_data.index)])
        with open("Results/" + name + "/" + file, "r") as f:
            res = json.load(f)
        multimodal[9].append(res["nn_f1_score"]-base)
    for file in [x for x in list(os.listdir("Results/" + name + "/")) if x.endswith("cmr.json")]:
        with open("Problems/" + name + "/expert_theories/" + file[:7] + "_expert.json") as f:
            problem = json.load(f)
            expert = load_theory(problem["expert_query"])
            test_data = data.loc[problem["test_indices"]]
            covered = expert.get_covering_examples(test_data)
            base = 0#f1_score(problem["test_labels"], [x in covered for x in list(test_data.index)])
        with open("Results/" + name + "/" + file, "r") as f:
            res = json.load(f)
        CMR[9].append(res["f1_score"]-base)

print(nether)
print(CMR)
print(NN)
print(multimodal)

# print([x/len(exps) for x in nether])
# print([x/len(exps) for x in CMR])
# print([x/len(exps) for x in NN])
# print([x/len(exps) for x in multimodal])
# plt.style.use("Experiments/tex.mplstyle")
# plt.figure(figsize=(3.31,2.04))
# plt.plot([10,20,30,40,50,60,70,80,90,100],[x/len(exps) for x in nether], label="nether", marker="o")
# plt.plot([10,20,30,40,50,60,70,80,90,100],[x/len(exps) for x in CMR], label="CMR", marker="s")
#
# plt.plot([10,20,30,40,50,60,70,80,90,100],[x/len(exps) for x in NN], label="NN", marker="v")
# plt.plot([10,20,30,40,50,60,70,80,90,100],[x/len(exps) for x in multimodal], label="MLP+NC", marker=1)
#
#
# plt.legend()
# plt.xlabel("% Training data")
# plt.ylabel("Average F$_1$-score")
# plt.ylim([0,1])
# plt.show()
