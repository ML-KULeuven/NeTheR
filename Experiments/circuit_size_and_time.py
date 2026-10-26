import json
import os
import matplotlib.pyplot as plt
from Data.dataloaders import load_data
from sklearn.metrics import f1_score
from src.propositional import load_theory
import numpy as np
res={}
m_star=[]
m_prime=[]
num_mods=[]
comp_size_times={}
train_size_times={}


def mean(x):
    if len(x)>0:
     return sum(x)/len(x)
    else:
        return 0
full_time=[]
full_k=[]
generate_mod_size_times={}
for name in ["animals","motifs","faces","birds", "medical","music","light", "objects", "housing"]:
    # data, _, _ = load_data(name)
    for width in range(2, 3):
        for depth in range(2, 5):
            for missing_var in range(1, 5):
                for it in range(0, 2):
                    with open("Problems/"+name+"/expert_theories/" + str(depth) + "_" + str(width) + "_" + str(missing_var) + "_" + str(it) + "_expert.json") as f:
                        problem=json.load(f)
                    print("Problems/"+name+"/expert_theories/" + str(depth) + "_" + str(width) + "_" + str(missing_var) + "_" + str(it) + "_expert.json")
                    m_star.append(len(load_theory(problem["original_query"]).nodes()))
                    prime_nodes=len(load_theory(problem["expert_query"]).nodes())
                    m_prime.append(len(load_theory(problem["expert_query"]).nodes()))
                    num_mods.append(2*len([x for x in load_theory(problem["expert_query"]).nodes() if x.is_var()])+len([x for x in load_theory(problem["expert_query"]).nodes() if not x.is_var()]))
                    all_files=[x for x in os.listdir("Results/"+name+"/")]
                    for file in all_files:
                        splitted=file.split("_")
                        if file.startswith(str(depth) + "_" + str(width) + "_" + str(missing_var) + "_" + str(it)+"_dt_paths.json") or file.endswith("True_True_paths.json"):
                        # if file.endswith("cmr.json"):
                            with open("Results/" + name + "/" + file, "r") as f:
                                result=json.load(f)
                            print(file)
                            full_time.append(result["time"])
                            full_k.append(result["full_timing"]["predicting_k"][0])
                            if prime_nodes not in comp_size_times:
                                comp_size_times[prime_nodes]=[]
                                train_size_times[prime_nodes] = []
                                generate_mod_size_times[prime_nodes] = []

                            comp_size_times[prime_nodes].append(sum(result["full_timing"]["constructing_model"]))

                            train_size_times[prime_nodes].append(sum(result["full_timing"]["training_neural_net"]))
                            generate_mod_size_times[prime_nodes].append(sum(result["full_timing"]["generating_modifications"]))

full_comp=[]
for x in comp_size_times:
    full_comp+=comp_size_times[x]
full_train=[]
for x in train_size_times:
    full_train+=train_size_times[x]
full_gen=[]
for x in generate_mod_size_times:
    full_gen+=generate_mod_size_times[x]

print("COMP: ", mean(full_comp)/60)
print("TRAIN: ", mean(full_train)/60)
print("GEN: ", mean(full_gen)/60)
print("K: ", mean(full_k)/60)
print("Total: ", mean(full_time)/60)
# all_times=[]
# for x in res:
#     all_times+=res[x]
# print("Mean: ", sum(all_times)/len(all_times)/60)
# plt.scatter([x for x in res],[sum(res[x])/len(res[x]) for x in res])
# plt.show()
print( min(num_mods),max(num_mods))
print("M_star: ", min(m_star),max(m_star),mean(m_star),np.std(m_star))
print("M_prime: ", min(m_prime),max(m_prime),mean(m_prime),np.std(m_prime))
# plt.plot(sorted([x for x in comp_size_times]),[mean(comp_size_times[x]) for x in sorted([x for x in comp_size_times])])
# plt.xlabel("circuit size M'")
# plt.ylabel("time (s)")
# plt.savefig("plots/comp_time.png")
# plt.clf()
# plt.plot(sorted([x for x in comp_size_times]),[mean(train_size_times[x]) for x in sorted([x for x in comp_size_times])])
# plt.xlabel("circuit size M'")
# plt.ylabel("time (s)")
# plt.savefig("plots/train_time.png")
# plt.clf()
# plt.plot(sorted([x for x in comp_size_times]),[mean(generate_mod_size_times[x]) for x in sorted([x for x in comp_size_times])])
# plt.xlabel("circuit size M'")
# plt.ylabel("time (s)")
# plt.savefig("plots/generating_mods_time.png")