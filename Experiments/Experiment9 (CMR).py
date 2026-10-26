from Data.dataloaders import load_data
from src.train_model import train_model, validate_model
from src.neural_architectures.convolutional import res_cnn, multimodal_cnn
from src.neural_architectures.mlp import mlp
from sklearn.metrics import f1_score, precision_score, recall_score
from src.propositional import load_theory
from src.construct_model import prop2sdd
from pysdd.sdd import SddManager,Vtree,WmcManager, SddNode
from src.Main import approximate_revision
from src.propositional import load_theory
import json
import time
import os
import torch
from CMR.cmr_src import CMR,netherEncoder

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

for name in ["motifs"]:
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
        ### EXPERIMENT 9: Multimodal NN
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


                        if False or not os.path.isfile("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_cmr.json"):
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
                                noncat_cols=list(full_data.columns)[-3*64*64:]
                            elif mode == "light_conv":
                                new_cols = used_columns + list(full_data.columns)[-1024:]
                                noncat_cols = list(full_data.columns)[-1024:]
                            elif mode == "motif_conv":
                                new_cols = used_columns + list(full_data.columns)[-510:]
                                noncat_cols = list(full_data.columns)[-510:]
                            else:
                                new_cols=used_columns+list(full_data.columns)[-518:]
                                noncat_cols = list(full_data.columns)[-518:]
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


                            counter=0
                            for cond in conditions:
                                if "==" in cond:
                                    training_data["c"+str(counter)]=training_data[cond[:cond.find("==")].strip()] == float(cond[cond.find("==") + 2:].strip())
                                    validation_data["c" + str(counter)] = validation_data[cond[:cond.find("==")].strip()] == float(cond[cond.find("==") + 2:].strip())
                                    test_data["c" + str(counter)] = test_data[cond[:cond.find("==")].strip()] == float(cond[cond.find("==") + 2:].strip())

                                elif ">" in cond:
                                    training_data["c" + str(counter)] = training_data[cond[:cond.find(">")].strip()] > float(cond[cond.find(">") + 1:].strip())
                                    validation_data["c" + str(counter)] = validation_data[cond[:cond.find(">")].strip()] > float(cond[cond.find(">") + 1:].strip())
                                    test_data["c" + str(counter)] = test_data[cond[:cond.find(">")].strip()] > float(cond[cond.find(">") + 1:].strip())

                                else:
                                    training_data["c" + str(counter)] = training_data[cond[:cond.find("<")].strip()] < float(cond[cond.find("<") + 1:].strip())
                                    validation_data["c" + str(counter)] = validation_data[cond[:cond.find("<")].strip()] < float(cond[cond.find("<") + 1:].strip())
                                    test_data["c" + str(counter)] = test_data[cond[:cond.find("<")].strip()] < float(cond[cond.find("<") + 1:].strip())

                                counter+=1

                            training_data=training_data[["c"+str(x) for x in range(0,counter)]+noncat_cols]
                            validation_data = validation_data[["c" + str(x) for x in range(0, counter)] + noncat_cols]
                            test_data = test_data[["c" + str(x) for x in range(0, counter)] + noncat_cols]

                            # print(training_data)
                            print("---- CMR ----")
                            nn_start=time.time()
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
                            cmr_rules=torch.tensor(rules)
                            print("Rules: ", cmr_rules.shape)

                            cmr=CMR(netherEncoder(mode), 512,200,1,20,w_y=10, n_concepts=len(var_dict),concept_names="")
                            # print("Rules shape: ", cmr_rules.shape)
                            cmr.add_rules_irr(cmr_rules)

                            train_model(cmr, (training_data,  validation_data),(training_labels, validation_labels), epochs, True)
                            nn_finish=time.time()
                            print("Validate model")
                            nn_prediction=validate_model(cmr, test_data, nn=mode=="animal_conv", cmr=True)
                            result["f1_score"] = f1_score(test_labels, nn_prediction)
                            print("----------> NEW RESULT: " ,result["f1_score"])
                            results.add(result["f1_score"])
                            result["precision_score"] = precision_score(test_labels, nn_prediction)
                            result["recall_score"] = recall_score(test_labels, nn_prediction)
                            result["prediction"] = nn_prediction
                            result["time"] = nn_finish - nn_start
                            with open("Results/"+name+"/"+str(depth)+"_"+str(width)+"_"+str(missing_var)+"_"+str(it)+"_cmr.json","w") as f:
                                json.dump(result, f)