from threnpi.Data.dataloaders import load_data
from threnpi.src.train_model import train_model, validate_model
from threnpi.src.neural_architectures.convolutional import res_cnn
from sklearn.metrics import f1_score, precision_score, recall_score
from threnpi.src.Main import approximate_revision
from threnpi.src.propositional import load_theory
import json
import time
import os

#EXPERIMENT 3: freeze, #edits = #removals, Min likelihood instead of Sharpe
name="faces"
full_data, cat_cols, drop_cols=load_data(name)
if name=="music":
    mode="music_conv"
else:
    mode="animal_conv"
print("DATA LOADED")
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
                edit_distance = len(original_theory.variables()) - len(
                    expert_theory.variables())  # how many edits are ideally needed

                k = 0  # alpha estimation
                alpha=0.99 #Alpha is useless here
                h = 1  # no hill climbing
                p = 5  # patience
                merge = False  # does nothing right now
                freeze = True  # previous changes are "locked"

                epochs = 25
                if not os.path.isfile("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(k)+"_"+str(h)+"_"+str(p)+"_"+str(edit_distance)+"_"+str(merge)+"_"+str(freeze)+"_min.json"):
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
                    best_result=approximate_revision(input_data, expert_theory,input_labels, edit_distance, k,h,p,merge, mode,freeze,alpha=alpha, epochs=epochs, estimate="min")
                    finish=time.time()
                    if best_result[1]==None: #only removals
                        covered_test_indices=best_result[2].get_covering_examples(test_data)
                        prediction=[x in covered_test_indices for x in test_data.index]
                    else:
                        prediction=validate_model(best_result[1], test_data)
                    print("---- WRITING RESULTS ----")
                    result={}
                    result["f1_score"]=f1_score(test_labels, prediction)
                    result["precision_score"] = precision_score(test_labels, prediction)
                    result["recall_score"] = recall_score(test_labels, prediction)
                    result["original_theory"]=original_theory.save()
                    result["revised_theory"] = best_result[2].save()
                    result["prediction"]=prediction
                    result["time"]=finish-start
                    with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(k)+"_"+str(h)+"_"+str(p)+"_"+str(edit_distance)+"_"+str(merge)+"_"+str(freeze)+"_min.json","w") as f:
                        json.dump(result, f)