
import json
from threnpi.src.propositional import load_theory
import os
from threnpi.Data.dataloaders import load_small_animal_data
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

subresults={}
for file in os.listdir("Results/music/"):
    #only take freeze/not freeze
    splitted=file.split("_")
    if (len(splitted)==10 and splitted[-1]=="True.json") or (len(splitted)==11 and splitted[-1] in ["max.json", "min.json", "mean.json"]):
        # print(file)
        with open("Results/music/"+file) as f:
            result=json.load(f)


        if str(splitted[:4]) not in subresults:
            subresults[str(splitted[:4])]=[0,0,0,0]
        if len(splitted)==10:
            subresults[str(splitted[:4])][0]=result["f1_score"]
            if result["f1_score"]==0.0:
                print(file)
        elif splitted[-1]=="min.json":
            subresults[str(splitted[:4])][1] = result["f1_score"]
        elif splitted[-1]=="mean.json":
            subresults[str(splitted[:4])][2] = result["f1_score"]
        elif splitted[-1]=="max.json":
            query = result["revised_theory"]
            subresults[str(splitted[:4])][3] = result["f1_score"]
for res in subresults:
    print(res, subresults[res])
results=[]
for i in subresults:
    results.append(subresults[i])
# results=[x for x in results if 0 not in x]

results.sort()
sharpe=[x[0] for x in results]
min_res=[x[1] for x in results]
mean_res=[x[2] for x in results]
max_res=[x[3] for x in results]
print(len(sharpe))
print("Sharpe: ", sharpe)
print("Min: ", min_res)
print("Max: ", max_res)
print("Sharpe:", sum(sharpe)/len(sharpe))
print("Min:",sum(min_res)/len(min_res))
print("Mean:",sum(mean_res)/len(sharpe))
print("Max:",sum(max_res)/len(sharpe))
plt.scatter(list(range(0,len(sharpe))), list(sharpe), label="sharpe")
plt.scatter(list(range(0,len(min_res))), list(min_res), label="Min")
plt.scatter(list(range(0,len(mean_res))), list(mean_res), label="Mean")
plt.scatter(list(range(0,len(max_res))), list(max_res), label="Max")
plt.xlabel("Instantie ID")
plt.ylabel("f1-score")
plt.ylim([0,1.1])
plt.legend()
plt.savefig("plots/music_heuristics.png")
# with open("Problems/animals/expert_theories/3_2_2_0_expert.json") as f:
#     problem=json.load(f)
# print(load_theory(problem["original_query"]).show())
# print("---------")
# print(load_theory(problem["expert_query"]).show())