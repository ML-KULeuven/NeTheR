
import json
from threnpi.src.propositional import load_theory
import os
from threnpi.Data.dataloaders.animal_loader import load_small_animal_data
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

results=[]
full_data, _, _=load_small_animal_data()
for file in os.listdir("Results/"):
    if file.split("_")[-1]=="True.json":
        print(file)
        with open("Problems/animals/expert_theories/"+file[:7]+"_expert.json") as f:
            problem=json.load(f)

        test_indices = problem["test_indices"]
        test_labels = problem["test_labels"]
        test_data = full_data.loc[test_indices].copy()

        with open("Results/"+file) as f:
            result=json.load(f)
        expert_indices=load_theory(problem["expert_query"]).get_covering_examples(test_data)
        expert_labels=[x in expert_indices for x in test_data.index]
        results.append((float(f1_score(test_labels, expert_labels)),float(result["nn_f1_score"]),float(result["f1_score"])))
results.sort()
original_f1s=[x[0] for x in results]
nn_f1s=[x[1] for x in results]
rev_f1s=[x[2] for x in results]

plt.scatter(list(range(0,len(original_f1s))), list(original_f1s), label="Input")
plt.scatter(list(range(0,len(nn_f1s))), list(nn_f1s), label="Neural Net")
plt.scatter(list(range(0,len(rev_f1s))), list(rev_f1s), label="Revised")
plt.xlabel("Instantie ID")
plt.ylabel("f1-score")
plt.legend()
plt.show()
# with open("Problems/animals/expert_theories/3_2_2_0_expert.json") as f:
#     problem=json.load(f)
# print(load_theory(problem["original_query"]).show())
# print("---------")
# print(load_theory(problem["expert_query"]).show())