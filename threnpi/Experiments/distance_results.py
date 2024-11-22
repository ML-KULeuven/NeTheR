
import json
from threnpi.src.propositional import load_theory
import os
from threnpi.Data.dataloaders import load_small_animal_data
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

subresults={}
counter=0
for file in os.listdir("Results/animals/"):
    #only take freeze/not freeze
    splitted=file.split("_")
    if len(splitted)==10 and splitted[-1]=="True.json":
        print(file)
        counter+=1
        with open("Results/animals/"+file) as f:
            result=json.load(f)
        if str(splitted[2]) not in subresults:
            subresults[str(splitted[2])]=[0,0,0,0]
        else:
            subresults[str(splitted[2])][int(splitted[7])-1]+=result["f1_score"]

for k in subresults:
    plt.plot(list(range(0,len(subresults[k]))), [16*x/counter for x in subresults[k]], label=k)

plt.xlabel("Instantie ID")
plt.ylabel("f1-score")
plt.ylim([0,1.1])
plt.legend()
plt.show()
# plt.savefig("plots/distance.png")
# with open("Problems/animals/expert_theories/3_2_2_0_expert.json") as f:
#     problem=json.load(f)
# print(load_theory(problem["original_query"]).show())
# print("---------")
# print(load_theory(problem["expert_query"]).show())