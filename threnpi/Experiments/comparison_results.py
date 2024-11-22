
import json
from threnpi.src.propositional import load_theory
import os
from threnpi.Data.dataloaders import load_small_animal_data, load_housing_data, load_music_data
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

data, _,_=load_music_data()
subresults={}
baseline={}
for file in os.listdir("Results/music/"):
    splitted=file.split("_")
    if (len(splitted)==10 and splitted[-1]=="True.json") or (len(splitted)==11 and splitted[-1] in ["andor.json", "orand.json"]) or len(splitted)==5:
        print(file)
        with open("Results/music/"+file) as f:
            result=json.load(f)


        if str(splitted[:4]) not in subresults:

            with open("Problems/music/expert_theories/" + file[:7] + "_expert.json") as f:
                problem = json.load(f)
            expert = load_theory(problem["expert_query"])
            test_data=data.loc[problem["test_indices"]]
            covered=expert.get_covering_examples(test_data)
            subresults[str(splitted[:4])]=[0,0,0,0,0,f1_score(problem["test_labels"],[x in covered for x in set(test_data.index)])]
        if len(splitted)==10:
            subresults[str(splitted[:4])][0]=result["f1_score"]
            subresults[str(splitted[:4])][3] = result["nn_f1_score"]
        elif len(splitted)==5:
            subresults[str(splitted[:4])][4] = result["nn_f1_score"]
        elif splitted[-1]=="andor.json":
            subresults[str(splitted[:4])][1] = result["f1_score"]
        elif splitted[-1]=="orand.json":
            subresults[str(splitted[:4])][2] = result["f1_score"]

print("SUBR: ", len(subresults))
results=[]
for i in subresults:
    results.append(subresults[i])

results.sort()
sharpe=[x[0] for x in results]
andor=[x[1] for x in results]
orand=[x[2] for x in results]
nn=[x[3] for x in results]
mnn=[x[4] for x in results]
base=[x[5] for x in results]
print(len(sharpe))
print(sharpe)
print("Threnpi:", sum(sharpe)/len(sharpe))
print("And or:",sum(andor)/len(andor))
print("Or and:",sum(orand)/len(sharpe))
print("NN: ", sum(nn)/len(nn))
print("Multimodal nn: ", sum(mnn)/len(mnn))
print("Input: ", sum(base)/len(base))

plt.scatter(list(range(0,len(sharpe))), list(sharpe), label="Threnpi (Ours)")
plt.scatter(list(range(0,len(andor))), list(andor), label="And or")
plt.scatter(list(range(0,len(orand))), list(orand), label="Or and")
plt.scatter(list(range(0,len(nn))), list(nn), label="NN")
plt.scatter(list(range(0,len(mnn))), list(mnn), label="Multimodal NN")
plt.scatter(list(range(0,len(base))), list(base), label="Input")

plt.xlabel("Instantie ID")
plt.ylabel("f1-score")
plt.ylim([0,1.1])
plt.legend()
plt.savefig("plots/music_comparison.png")
# with open("Problems/animals/expert_theories/3_2_2_0_expert.json") as f:
#     problem=json.load(f)
# print(load_theory(problem["original_query"]).show())
# print("---------")
# print(load_theory(problem["expert_query"]).show())