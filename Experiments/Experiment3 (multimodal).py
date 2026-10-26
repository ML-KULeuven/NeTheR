from Data.dataloaders import load_data
from src.train_model import train_model, validate_model
from src.neural_architectures.convolutional import res_cnn, multimodal_cnn
from src.neural_architectures.Inception import multimodal_inception
from src.neural_architectures.mlp import mlp
from sklearn.metrics import f1_score, precision_score, recall_score
from src.Main import approximate_revision
from src.propositional import load_theory
import json
import time
import os


name="motifs"
full_data, cat_cols, drop_cols=load_data(name)
if name == "music":
    mode = "music_conv"  # used to determine shape of the neural predicate
elif name == "light":
    mode = "light_conv"
elif name == "motifs":
    mode = "motif_conv"
else:
    mode = "animal_conv"
results=set([0])
while 0 in results:
    results=set()
    ### EXPERIMENT 3: Multimodal NN
    for width in range(2, 3):
        for depth in range(2, 5):
            for missing_var in range(1, 5):
                for it in range(0, 2):
                    with open("Problems/"+name+"/expert_theories/" + str(depth) + "_" + str(width) + "_" + str(
                            missing_var) + "_" + str(it) + "_expert.json") as f:
                        problem = json.load(f)
                        expert_theory = load_theory(problem["expert_query"])
                        original_theory = load_theory(problem["original_query"])
                    epochs=100


                    if not os.path.isfile("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_multimodal.json"):
                        check=False
                        if os.path.isfile("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_multimodal.json"):
                            with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_multimodal.json", "r") as f:
                                test=json.load(f)
                            if test["nn_f1_score"]>0:
                                check=True
                        if not check:
                            print("@@@@@@@@@@@@@@@@@@@@ NEW PROBLEM: ", depth,width,  missing_var, it)
                            conditions = [x.condition for x in expert_theory.variables()]
                            used_columns = set()
                            for cond in conditions:
                                if "==" in cond:
                                    used_columns.add(cond[:cond.find("==")].strip())
                                elif ">" in cond:
                                    used_columns.add(cond[:cond.find(">")].strip())
                                else:
                                    used_columns.add(cond[:cond.find("<")].strip())
                            used_columns=list(used_columns)
                            if mode=="animal_conv":
                                new_cols = used_columns+list(full_data.columns)[-3*64*64:]
                            elif mode=="light_conv":
                                new_cols = used_columns + list(full_data.columns)[-1024:]
                            elif mode=="motif_conv":
                                new_cols = used_columns + list(full_data.columns)[-510:]
                            else:
                                new_cols=used_columns+list(full_data.columns)[-518:]
                            training_indices=problem["train_indices"]
                            training_labels=problem["train_labels"]
                            training_data=full_data.loc[training_indices].copy()[new_cols]
                            validation_indices = problem["validation_indices"]
                            validation_labels = problem["validation_labels"]
                            validation_data = full_data.loc[validation_indices].copy()[new_cols]
                            test_indices = problem["test_indices"]
                            test_labels = problem["test_labels"]
                            test_data = full_data.loc[test_indices].copy()[new_cols]
                            input_data=(training_data, validation_data)
                            input_labels=(training_labels, validation_labels)
                            result={}
                            print("---- NEURAL NET ----")
                            nn_start=time.time()
                            if mode=="animal_conv":
                                neural_net=multimodal_cnn(len(new_cols)-3*64*64,(len(training_labels)-sum(training_labels))/len(training_labels))
                            elif mode=="light_conv":
                                neural_net=multimodal_inception(len(new_cols)-1024,(len(training_labels)-sum(training_labels))/len(training_labels))
                            elif mode == "motif_conv":
                                neural_net = multimodal_inception(len(new_cols) - 510,
                                                                  (len(training_labels) - sum(training_labels)) / len(
                                                                      training_labels), input_dim=510)

                            else:
                                neural_net=mlp((len(training_labels)-sum(training_labels))/len(training_labels),len(new_cols))
                            train_model(neural_net, (training_data,  validation_data),(training_labels, validation_labels), epochs, True)
                            nn_finish=time.time()
                            nn_prediction=validate_model(neural_net, test_data, nn=mode!="music_conv")
                            result["nn_f1_score"] = f1_score(test_labels, nn_prediction)
                            print("----------> NEW RESULT: " ,result["nn_f1_score"])
                            results.add(result["nn_f1_score"])
                            result["nn_precision_score"] = precision_score(test_labels, nn_prediction)
                            result["nn_recall_score"] = recall_score(test_labels, nn_prediction)
                            result["nn_prediction"] = nn_prediction
                            result["nn_time"] = nn_finish - nn_start
                            with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_multimodal.json","w") as f:
                                json.dump(result, f)