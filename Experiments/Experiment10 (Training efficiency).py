from Data.dataloaders import load_data
from src.train_model import train_model, validate_model
from src.neural_architectures.convolutional import res_cnn
from src.neural_architectures.mlp import mlp
from src.neural_architectures.Inception import inception
from sklearn.metrics import f1_score, precision_score, recall_score
from src.Main import approximate_revision
from src.propositional import load_theory
import json
from src.neural_architectures.convolutional import res_cnn, multimodal_cnn
from src.neural_architectures.Inception import multimodal_inception
from sklearn.model_selection import train_test_split
import time
import os
import torchtime.models as models
import torch
from CMR.cmr_src import CMR,netherEncoder
from pysdd.sdd import SddManager,Vtree,WmcManager, SddNode
from src.construct_model import prop2sdd

def merge_pmodels(pmodel1, pmodel2):
    """
    Merge two partial models.
    If one model is None, it represents False and the union returns False.
    """
    if pmodel1 is None or pmodel2 is None:
        return None
    return {**pmodel1, **pmodel2}

def partial_models(f: SddNode):
    """A generator for the models of an SDD."""
    if f.is_decision():
        for prime, sub in f.elements():
            # iterate over all dicts of prime
            # iterate over all dicts of sub
            # yield union of dicts
            for pmodel in partial_models(prime):
                for smodel in partial_models(sub):
                    yield merge_pmodels(pmodel, smodel)
    elif f.is_literal():
        yield {abs(f.literal): 0 if f.literal < 0 else 1}
    elif f.is_true():
        yield {}
    elif f.is_false():
        return None
    else:
        raise ValueError("Unknown SDD node type")


for name in ["motifs",]:# "faces","birds", "medical","music","light", "objects"]:
    full_data, cat_cols, drop_cols=load_data(name)
    for percentage in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
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
                        h = 1  # no hill climbing
                        p = 5  # patience
                        merge = False  # does nothing right now
                        freeze = True  # previous changes are "locked"
                        if name=="music":
                            mode = "music_conv"  # used to determine shape of the neural predicate
                        elif name=="light":
                            mode="light_conv"
                        elif name == "motifs":
                            mode = "motif_conv"
                        else:
                            mode="animal_conv"
                        epochs = 25
                        already_done=False
                        if os.path.isfile(
                                "Results/" + name + "/" + str(depth) + "_" + str(width) + "_" + str(
                                    missing_var) + "_" + str(it) + "_training_" + str(percentage) + ".json"):
                            with open(
                                    "Results/" + name + "/" + str(depth) + "_" + str(width) + "_" + str(
                                        missing_var) + "_" + str(it) + "_training_" + str(percentage) + ".json") as f:
                                existing = json.load(f)
                                if "cmr_f1_score" in existing:
                                    already_done = True
                        if not already_done or not os.path.isfile("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_training_"+str(percentage)+".json"):
                            print("@@@@@@@@@@@@@@@@@@@@ NEW PROBLEM: ",name, depth,width,  missing_var, it, percentage)
                            training_indices=problem["train_indices"]
                            training_labels=problem["train_labels"]
                            training_data=full_data.loc[training_indices].copy()
                            training_data, _, training_labels, _ = train_test_split(training_data,training_labels,test_size=1-percentage)
                            validation_indices = problem["validation_indices"]
                            validation_labels = problem["validation_labels"]
                            validation_data = full_data.loc[validation_indices].copy()
                            test_indices = problem["test_indices"]
                            test_labels = problem["test_labels"]
                            test_data = full_data.loc[test_indices].copy()



                            input_data=(training_data, validation_data)
                            input_labels=(training_labels, validation_labels)


                            start=time.time()
                            best_result=approximate_revision(input_data, expert_theory,input_labels, edit_distance, k,h,p,merge, mode,freeze, epochs=epochs)
                            finish=time.time()
                            if best_result[1]==None: #only removals
                                covered_test_indices=best_result[2].get_covering_examples(test_data)
                                prediction=[x in covered_test_indices for x in test_data.index]
                            else:
                                print("Prediction on test data")
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
                            print("---- NEURAL NET ----")
                            nn_start=time.time()
                            if mode=="animal_conv":
                                neural_net=res_cnn((len(training_labels)-sum(training_labels))/len(training_labels))
                            elif mode=="light_conv":
                                neural_net=inception((len(training_labels)-sum(training_labels))/len(training_labels))
                            elif mode=="motif_conv":
                                neural_net=inception((len(training_labels)-sum(training_labels))/len(training_labels), input_dim=510)
                            else:
                                neural_net=mlp((len(training_labels)-sum(training_labels))/len(training_labels))
                            train_model(neural_net, (training_data,  validation_data),(training_labels, validation_labels), epochs, True)
                            nn_finish=time.time()
                            nn_prediction=validate_model(neural_net, test_data, nn=mode=="animal_conv")
                            result["nn_f1_score"] = f1_score(test_labels, nn_prediction)
                            result["nn_precision_score"] = precision_score(test_labels, nn_prediction)
                            result["nn_recall_score"] = recall_score(test_labels, nn_prediction)
                            result["nn_prediction"] = nn_prediction
                            result["nn_time"] = nn_finish - nn_start

                            print("---- Multimodal ----")
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
                            if mode == "animal_conv":
                                new_cols = used_columns + list(full_data.columns)[-3 * 64 * 64:]
                                noncat_cols = list(full_data.columns)[-3 * 64 * 64:]
                            elif mode == "light_conv":
                                new_cols = used_columns + list(full_data.columns)[-1024:]
                                noncat_cols = list(full_data.columns)[-1024:]
                            elif mode == "motif_conv":
                                new_cols = used_columns + list(full_data.columns)[-510:]
                                noncat_cols = list(full_data.columns)[-510:]
                            else:
                                new_cols = used_columns + list(full_data.columns)[-518:]
                                noncat_cols = list(full_data.columns)[-518:]
                            nn_start = time.time()
                            if mode == "animal_conv":
                                neural_net = multimodal_cnn(len(new_cols) - 3 * 64 * 64,
                                                            (len(training_labels) - sum(training_labels)) / len(
                                                                training_labels))
                            elif mode == "light_conv":
                                neural_net = multimodal_inception(len(new_cols) - 1024,
                                                                  (len(training_labels) - sum(training_labels)) / len(
                                                                      training_labels))
                            elif mode == "motif_conv":
                                neural_net = multimodal_inception(len(new_cols) - 510,
                                                                  (len(training_labels) - sum(training_labels)) / len(
                                                                      training_labels), input_dim=510)
                            else:
                                neural_net = mlp((len(training_labels) - sum(training_labels)) / len(training_labels),
                                                 len(new_cols))
                            training_data=training_data[new_cols]
                            validation_data=validation_data[new_cols]
                            test_data=test_data[new_cols]

                            train_model(neural_net, (training_data, validation_data), (training_labels, validation_labels),
                                        epochs, True)
                            nn_finish = time.time()
                            nn_prediction = validate_model(neural_net, test_data, nn=mode != "music_conv")
                            result["multi_f1_score"] = f1_score(test_labels, nn_prediction)

                            result["multi_precision_score"] = precision_score(test_labels, nn_prediction)
                            result["multi_recall_score"] = recall_score(test_labels, nn_prediction)
                            result["multi_prediction"] = nn_prediction
                            result["multi_time"] = nn_finish - nn_start

                            counter = 0
                            for cond in conditions:
                                if "==" in cond:
                                    training_data["c" + str(counter)] = training_data[
                                                                            cond[:cond.find("==")].strip()] == float(
                                        cond[cond.find("==") + 2:].strip())
                                    validation_data["c" + str(counter)] = validation_data[
                                                                              cond[:cond.find("==")].strip()] == float(
                                        cond[cond.find("==") + 2:].strip())
                                    test_data["c" + str(counter)] = test_data[cond[:cond.find("==")].strip()] == float(
                                        cond[cond.find("==") + 2:].strip())

                                elif ">" in cond:
                                    training_data["c" + str(counter)] = training_data[
                                                                            cond[:cond.find(">")].strip()] > float(
                                        cond[cond.find(">") + 1:].strip())
                                    validation_data["c" + str(counter)] = validation_data[
                                                                              cond[:cond.find(">")].strip()] > float(
                                        cond[cond.find(">") + 1:].strip())
                                    test_data["c" + str(counter)] = test_data[cond[:cond.find(">")].strip()] > float(
                                        cond[cond.find(">") + 1:].strip())

                                else:
                                    training_data["c" + str(counter)] = training_data[
                                                                            cond[:cond.find("<")].strip()] < float(
                                        cond[cond.find("<") + 1:].strip())
                                    validation_data["c" + str(counter)] = validation_data[
                                                                              cond[:cond.find("<")].strip()] < float(
                                        cond[cond.find("<") + 1:].strip())
                                    test_data["c" + str(counter)] = test_data[cond[:cond.find("<")].strip()] < float(
                                        cond[cond.find("<") + 1:].strip())

                                counter += 1

                            training_data = training_data[["c" + str(x) for x in range(0, counter)] + noncat_cols]
                            validation_data = validation_data[["c" + str(x) for x in range(0, counter)] + noncat_cols]
                            test_data = test_data[["c" + str(x) for x in range(0, counter)] + noncat_cols]

                            # print(training_data)
                            print("---- CMR ----")
                            nn_start = time.time()
                            # set up vtree and manager
                            theory = load_theory(problem["expert_query"])
                            query_vars = theory.variables() + theory.neural_predicates()

                            var_count = len(query_vars)
                            var_order = list(range(1, var_count + 1))
                            vtree_type = "balanced"

                            vtree = Vtree(var_count, var_order, vtree_type)
                            manager = SddManager.from_vtree(vtree)

                            var_list = [manager.literal(i) for i in range(1, var_count + 1)]
                            var_dict = {query_vars[x]: var_list[x] for x in range(0, len(query_vars))}

                            sdd_form = prop2sdd(theory, var_dict)
                            sdd_form.ref()
                            manager.minimize()
                            rules = []
                            for x in partial_models(sdd_form):
                                rule = []
                                for i in [y.literal for y in var_dict.values()]:
                                    if i not in x:
                                        rule.append([0, 0, 1])
                                    elif x[i] == 0:
                                        rule.append([0, 1, 0])
                                    else:
                                        rule.append([1, 0, 0])
                                rules.append(rule)
                            cmr_rules = torch.tensor(rules)

                            cmr = CMR(netherEncoder(mode), 512, 200, 1, 20, w_y=10, n_concepts=len(var_dict),
                                      concept_names="")
                            # print("Rules shape: ", cmr_rules.shape)
                            cmr.add_rules_irr(cmr_rules)

                            train_model(cmr, (training_data, validation_data), (training_labels, validation_labels),
                                        epochs, True)
                            nn_finish = time.time()
                            print("Validate model")
                            nn_prediction = validate_model(cmr, test_data, nn=mode == "animal_conv", cmr=True)
                            result["cmr_f1_score"] = f1_score(test_labels, nn_prediction)
                            print("----------> NEW RESULT: ", result["f1_score"])

                            result["cmr_precision_score"] = precision_score(test_labels, nn_prediction)
                            result["cmr_recall_score"] = recall_score(test_labels, nn_prediction)
                            result["cmr_prediction"] = nn_prediction
                            result["cmr_time"] = nn_finish - nn_start
                            with open("Results/" + name + "/" + str(depth) + "_" + str(width) + "_" + str(
                                    missing_var) + "_" + str(it) + "_training_"+str(percentage)+".json", "w") as f:
                                json.dump(result, f)
