from Data.dataloaders import load_data
from src.train_model import train_model, validate_model
from src.neural_architectures.convolutional import res_cnn
from src.neural_architectures.mlp import mlp
from src.neural_architectures.Inception import inception
from sklearn.metrics import f1_score, precision_score, recall_score
from src.Main import approximate_revision
from src.propositional import load_theory
from src.DT import learn_decision_tree
import json
import time
import os
import torchtime.models as models
import math

device=[0,]
for name in ["animals","motifs","faces","birds", "medical","music","light", "objects","housing"]:
    full_data, cat_cols, drop_cols=load_data(name)
    ### EXPERIMENT 1: freeze, #edits = #removed var
    for width in range(2, 3):
        for depth in range(2, 5):
            for missing_var in range(1, 5):
                for it in range(0, 2):
                    with open("Problems/"+name+"/expert_theories/" + str(depth) + "_" + str(width) + "_" + str(
                            missing_var) + "_" + str(it) + "_expert.json") as f:
                        problem = json.load(f)
                        expert_theory = load_theory(problem["expert_query"])
                        original_theory = load_theory(problem["original_query"])
                    # PARAMETERS
                    edit_distance = 4  # how many edits are ideally needed
                    k = 3  # alpha estimation

                    freeze = True  # previous changes are "locked"
                    if name=="music":
                        mode = "music_conv"  # used to determine shape of the neural predicate
                    elif name=="light":
                        mode="light_conv"
                    elif name=="motifs":
                        mode="motif_conv"
                    else:
                        mode="animal_conv"
                    epochs = 50
                    if not os.path.isfile("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_dt_paths.json"):
                        print("@@@@@@@@@@@@@@@@@@@@ NEW PROBLEM: ", depth,width,  missing_var, it)
                        training_indices=problem["train_indices"]
                        training_labels=problem["train_labels"]
                        training_data=full_data.loc[training_indices].copy()
                        validation_indices = problem["validation_indices"]
                        validation_labels = problem["validation_labels"]
                        validation_data = full_data.loc[validation_indices].copy()
                        test_indices = problem["test_indices"]
                        test_labels = problem["test_labels"]
                        test_data = full_data.loc[test_indices].copy()
                        input_data=(training_data, validation_data)
                        input_labels=(training_labels, validation_labels)
                        f1_dt=1.0
                        counter=0
                        while f1_dt>0.9 and len(todo)!=0:
                            counter+=1
                            split_size, leaf_size, tree_depth=todo.pop(0)
                            expert_theory, clf = learn_decision_tree(training_data.copy(), training_labels, cat_cols, split_size, leaf_size, tree_depth) #50 10 5
                            if expert_theory!=None:
                                covered = expert_theory.get_covering_examples(test_data)
                                f1_dt = f1_score(problem["test_labels"], [x in covered for x in list(test_data.index)])
                                if f1_dt<0.9:
                                    print("DECISION TREE: ", f1_dt, "Counter:", counter)


                            conditions = [x.condition for x in expert_theory.variables()]
                            used_columns = set()
                            for cond in conditions:
                                if "==" in cond:
                                    used_columns.add(cond[:cond.find("==")].strip())
                                elif ">" in cond:
                                    used_columns.add(cond[:cond.find(">")].strip())
                                else:
                                    used_columns.add(cond[:cond.find("<")].strip())
                            used_columns = list(used_columns)
                            start=time.time()
                            best_result, timing, operations_path=approximate_revision(input_data, expert_theory,input_labels,used_columns, edit_distance, k, mode,freeze, epochs=epochs, device=device,test_data=test_data, test_labels=test_labels)
                            finish=time.time()
                            if best_result[1]==None: #only removals
                                covered_test_indices=best_result[2].get_covering_examples(test_data)
                                prediction=[x in covered_test_indices for x in test_data.index]
                            else:
                                print("Prediction on test data")
                                prediction=validate_model(best_result[1], test_data)
                            print("---- WRITING RESULTS ----")
                            result={}
                            result["full_timing"]=timing
                            result["f1_score"]=f1_score(test_labels, prediction)
                            result["precision_score"] = precision_score(test_labels, prediction)
                            result["recall_score"] = recall_score(test_labels, prediction)
                            result["original_theory"]=original_theory.save()
                            result["decision_tree"]=expert_theory.save()
                            result["revised_theory"] = best_result[2].save()
                            result["prediction"]=prediction
                            result["operation_path"] = operations_path
                            result["time"]=finish-start

                            with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_dt_paths.json","w") as f:
                                json.dump(result, f)
                        else:
                            print("No DT available")

