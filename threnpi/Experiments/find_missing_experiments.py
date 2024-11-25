import os

for name in ["housing", "medical","faces","objects","birds","animals"]:
    all_files=list(os.listdir("Results/"+name+"/"))
    for width in range(2, 3):
        for depth in range(2, 5):
            for missing_var in range(1, 5):
                for it in range(0, 2):
                    spec_files=[x for x in all_files if x[:7]==str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)]
                    if len(spec_files)!=8:
                        todo = ["max.json", "mean.json", "min.json", "False.json", "True.json", "andor.json", "orand.json", "multimodal.json"]
                        for spec in spec_files:
                            todo.remove(spec.split("_")[-1])
                        for i in todo:
                            print(name, depth, width, missing_var, it, i)


