from threnpi.Data.dataloaders import load_data
from threnpi.src.train_model import train_model, validate_model
from threnpi.src.neural_architectures.convolutional import res_cnn
from sklearn.metrics import f1_score, precision_score, recall_score
from threnpi.src.Main import approximate_revision
from threnpi.src.propositional import load_theory, NEURAL_PREDICATE, OR, AND
import json
import time
import os
import random
from threnpi.src.construct_model import construct_model

name="faces"
full_data, cat_cols, drop_cols=load_data(name)
if name=="music":
    mode="music_conv"
else:
    mode="animal_conv"
### EXPERIMENT 7: (M' and N) or N
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
                h = 1  # no hill climbing
                p = 5  # patience
                merge = False  # does nothing right now
                freeze = True  # previous changes are "locked"

                epochs = 25
                if not os.path.isfile("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(k)+"_"+str(h)+"_"+str(p)+"_"+str(edit_distance)+"_"+str(merge)+"_"+str(freeze)+"_andor.json"):
                    print("@@@@@@@@@@@@@@@@@@@@ NEW PROBLEM: ", depth,width,  missing_var, it)
                    training_indices=problem["train_indices"]
                    train_labels=problem["train_labels"]
                    train_data=full_data.loc[training_indices].copy()
                    validation_indices = problem["validation_indices"]
                    validation_labels = problem["validation_labels"]
                    validation_data = full_data.loc[validation_indices].copy()
                    test_indices = problem["test_indices"]
                    test_labels = problem["test_labels"]
                    test_data = full_data.loc[test_indices].copy()
                    start=time.time()
                    random.seed(10)

                    pos_weight = (len(train_labels) - sum(train_labels)) / len(train_labels)

                    np1 = NEURAL_PREDICATE(max([-1] + [x.id for x in expert_theory.neural_predicates()]) + 1, mode)
                    np2 = NEURAL_PREDICATE(max([-1] + [x.id for x in expert_theory.neural_predicates()]) + 2, mode)

                    altered_theory= AND(max([-1] + [x.id for x in expert_theory.nodes() if x.is_and()]) + 1, [np2,OR(max([-1] + [x.id for x in expert_theory.nodes() if x.is_or()]) + 1, [expert_theory, np1])])
                    model = construct_model(altered_theory, train_data.columns, pos_weight, mode,
                                            freeze)
                    print("---- TRAINING ----")
                    train_model(model, (train_data, validation_data), (train_labels, validation_labels),
                                epochs, True)
                    finish=time.time()
                    prediction=validate_model(model, test_data)
                    print("---- WRITING RESULTS ----")
                    result={}
                    result["f1_score"]=f1_score(test_labels, prediction)
                    result["precision_score"] = precision_score(test_labels, prediction)
                    result["recall_score"] = recall_score(test_labels, prediction)
                    result["original_theory"]=original_theory.save()
                    result["revised_theory"] = altered_theory.save()
                    result["prediction"]=prediction
                    result["time"]=finish-start
                    with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_"+str(k)+"_"+str(h)+"_"+str(p)+"_"+str(edit_distance)+"_"+str(merge)+"_"+str(freeze)+"_andor.json","w") as f:
                        json.dump(result, f)