import os
import json
import numpy

counts=[0,0]
for path_id,path in enumerate(["True_paths", "dt_paths", "paths"]):
    setting=["Structurally informed", "sturcturally uninformed", "combined"][path_id]
    for id in [1,2]:
        testset=["Test", "Validation"][id-1]
        for data_id,datasets in enumerate([["music","motifs","light"],["housing","animals", "faces","birds", "medical", "objects"],["housing","animals", "faces","birds", "medical", "objects","music","motifs","light"]]):
            dataset=["Non-image","Image","All"][data_id]

            print("-------")
            print(setting,"|", testset,"|", dataset)
            symb = []
            neural = []
            rem = []
            for name in datasets:
                for i in [x for x in os.listdir(f"Results/{name}") if path in x]:
                    with open(f"Results/{name}/{i}") as f:
                        res = json.load(f)
                        if "edit_distance" in res:
                            if len(res["operation_path"])-1==res["edit_distance"]:
                                counts[0]+=1
                            else:
                                counts[1]+=1
                        local_symb = 0
                        local_neural = 0
                        local_rem = 0
                        prev = res["operation_path"][0][id]
                        for x in res["operation_path"][1:]:
                            if x[0][0] == "S":
                                local_symb += x[id] - prev
                            elif x[0][0] == "R":
                                local_rem += x[id] - prev
                            else:
                                local_neural += x[id] - prev
                            prev = x[id]
                        symb.append(local_symb)
                        neural.append(local_neural)
                        rem.append(local_rem)

            print("Symb: ", round(sum(symb) / len(symb),2),round(numpy.std(symb),2))
            print("Neural: ", round(sum(neural) / len(neural),2),round(numpy.std(neural),2))
            print("Rem: ", round(sum(rem) / len(rem),2),round(numpy.std(rem),2))

print("||||||||||||||||||")
print(counts)
