
import json
from threnpi.src.propositional import load_theory
import os
from matplotlib import pyplot as plt

subresults={}
for file in os.listdir("Results/music/"):
    #only take freeze/not freeze
    splitted=file.split("_")
    if len(splitted)==10 and splitted[2]==splitted[7]:
        print(file)
        with open("Results/music/"+file) as f:
            result=json.load(f)
        query=result["revised_theory"]
        if len([x for x in query if x[0]=="N"])>0:
            print(query)
        if str(splitted[:-1]) not in subresults:
            subresults[str(splitted[:-1])]=[0,0]
        if splitted[-1]=="True.json":
            subresults[str(splitted[:-1])][0]=result["f1_score"]
        else:
            subresults[str(splitted[:-1])][1] = result["f1_score"]

results=[]
for i in subresults:
    results.append(subresults[i])
results.sort()
freeze=[x[0] for x in results]
not_freeze=[x[1] for x in results]

plt.scatter(list(range(0,len(freeze))), list(freeze), label="Freeze")
plt.scatter(list(range(0,len(not_freeze))), list(not_freeze), label="No freeze")
plt.xlabel("Instantie ID")
plt.ylabel("f1-score")
plt.ylim([0,1.1])
plt.legend()
plt.savefig("plots/freeze_music.png")
# with open("Problems/animals/expert_theories/3_2_2_0_expert.json") as f:
#     problem=json.load(f)
# print(load_theory(problem["original_query"]).show())
# print("---------")
# print(load_theory(problem["expert_query"]).show())