from Data.dataloaders import load_data
from src.train_model import train_model, validate_model
from src.neural_architectures.convolutional import res_cnn
from src.neural_architectures.mlp import mlp
from src.neural_architectures.Inception import inception
from sklearn.metrics import f1_score, precision_score, recall_score
from src.Main import approximate_revision
from src.propositional import load_theory
import json
import time
import os

device=[1,]
for name in ["animals","motifs","faces","birds", "medical","music","light", "objects", "housing"]:
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
                    edit_distance = len(original_theory.variables()) - len(expert_theory.variables())  # how many edits are ideally needed
                    k = 3  # alpha estimation
                    freeze = False  # previous changes are "locked"
                    symb_allowed=False
                    if name=="music":
                        mode = "music_conv"  # used to determine shape of the neural predicate
                    elif name=="light":
                        mode="light_conv"
                    elif name=="motifs":
                        mode="motif_conv"
                    else:
                        mode="animal_conv"
                    epochs = 50
                    print("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(k)+"_"+str(freeze)+"_"+str(symb_allowed)+".json")
                    if not os.path.isfile("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(k)+"_"+str(freeze)+"_"+str(symb_allowed)+".json"):
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


                        start=time.time()
                        best_result, timing=approximate_revision(input_data, expert_theory,input_labels,cat_cols, edit_distance, k, mode,freeze, epochs=epochs, device=device, symb_allowed=symb_allowed)
                        finish=time.time()
                        if best_result[1]==None: #only symbolic
                            covered_test_indices=best_result[2].get_covering_examples(test_data)
                            prediction=[x in covered_test_indices for x in test_data.index]
                        else:
                            print("Prediction on test data")
                            prediction=validate_model(best_result[1], test_data, device=device)
                        print("---- WRITING RESULTS ----")
                        result={}
                        result["full_timing"]=timing
                        result["f1_score"]=f1_score(test_labels, prediction)
                        result["precision_score"] = precision_score(test_labels, prediction)
                        result["recall_score"] = recall_score(test_labels, prediction)
                        result["original_theory"]=original_theory.save()
                        result["revised_theory"] = best_result[2].save()
                        result["prediction"]=prediction
                        result["time"]=finish-start
                        print("---- NEURAL NET ----")
                        nn_start=time.time()
                        if mode=="animal_conv":
                            neural_net=res_cnn((len(training_labels)-sum(training_labels))/len(training_labels))
                        elif mode=="light_conv":
                            neural_net=inception((len(training_labels)-sum(training_labels))/len(training_labels))
                        elif mode == "motif_conv":
                            neural_net = inception((len(training_labels) - sum(training_labels)) / len(training_labels), input_dim=510)

                        else:
                            neural_net=mlp((len(training_labels)-sum(training_labels))/len(training_labels))
                        train_model(neural_net, (training_data,  validation_data),(training_labels, validation_labels), epochs, True, device=device)
                        nn_finish=time.time()
                        nn_prediction=validate_model(neural_net, test_data, nn=mode=="animal_conv", device=device)
                        result["nn_f1_score"] = f1_score(test_labels, nn_prediction)
                        result["nn_precision_score"] = precision_score(test_labels, nn_prediction)
                        result["nn_recall_score"] = recall_score(test_labels, nn_prediction)
                        result["nn_prediction"] = nn_prediction
                        result["nn_time"] = nn_finish - nn_start
                        with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(k)+"_"+str(freeze)+"_"+str(symb_allowed)+".json","w") as f:
                            json.dump(result, f)
