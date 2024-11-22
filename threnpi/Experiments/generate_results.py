from threnpi.Data.dataloaders.animal_loader import load_small_animal_data
import os
import json
from threnpi.src.propositional import query_prop
from threnpi.src.construct_model import flow_reconstruction_lightning
import copy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import precision_score
import pytorch_lightning as L
from sklearn.model_selection import train_test_split

from threnpi_old.Main.neural_architectures.convolutional import res_cnn
from threnpi.src.train_model import trainModel_lightning


full_data, categorical_columns, droppable_cols = load_small_animal_data()
full_data.iloc[:, -3 * 64 * 64:] = full_data.iloc[:, -3 * 64 * 64:] / 255

problem_name = "_small"
print(os.getcwd())
if True:
    for it in range(0, 10):
        for depth in range(2, 6):
            for missing_var in range(1, 5):
                for boil in [False, True]:
                    print("-------------",depth, missing_var, it, boil)
                    if boil:
                        result_name = "_small_boiled"
                    else:
                        result_name = "_small"
                    if os.path.exists("Problems/animals/" + str(depth) + "_" + str(
                            missing_var) + "_" + str(it) + problem_name + ".json"):
                        with open("Problems/animals/" + str(depth) + "_" + str(
                                missing_var) + "_" + str(it) + problem_name + ".json", "r") as f:
                            problem = json.load(f)
                        if os.path.exists("Results/animals/" + str(depth) + "_" + str(
                                missing_var) + "_" + str(it) + result_name + ".json"):
                            with open("Results/animals/" + str(depth) + "_" + str(
                                    missing_var) + "_" + str(it) + result_name + ".json", "r") as f:
                                result = json.load(f)
                        else:
                            result = {}
                        print(result)
                        original_query = problem["original_query"]
                        input_query = problem["input_query"]
                        dt_query = problem["DT"]
                        rule_query = problem["Rules"]
                        if type(rule_query["P"]) == str:
                            rule_query["P"] = [rule_query["P"]]
                        dropped_columns = problem["dropped_columns"]
                        train_indices = problem["train_indices"]
                        train_data = full_data.loc[train_indices]
                        test_data = full_data.loc[problem["test_indices"]].drop(columns=dropped_columns,
                                                                                errors='ignore')
                        train_labels = problem["train_labels"]
                        test_labels = problem["test_labels"]
                        pos_weight = (len(train_labels) - sum(train_labels)) / sum(train_labels)
                        # generate labels for all of the data
                        ground_truth = query_prop(original_query, "P", train_data)
                        ground_indices = set(ground_truth.index)
                        train_data = train_data.drop(columns=dropped_columns)

                        # #FLOW-reconstruction
                        if "flow_query" not in result:
                            print("-#-#-#-#-#-#INPUT")
                            print(original_query)
                            no_boil_query = {}
                            if boil:
                                no_boil_result = {}
                                with open("Results/animals/" + str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name.replace("_boiled", "") + ".json",
                                          "r") as f:
                                    no_boil_result = json.load(f)
                                no_boil_query = no_boil_result["flow_query"]
                            flow_query, model, boiled_effect, time = flow_reconstruction_lightning(train_data.copy(),
                                                                                             set(ground_indices.copy()),
                                                                                             copy.deepcopy(
                                                                                                 input_query),
                                                                                             dropped_columns.copy(),
                                                                                             test_data.copy(),
                                                                                             pos_weight,
                                                                                             "animal_conv", boil,
                                                                                             no_boil_query)
                            if (not boil) or (boil and boiled_effect):

                                np_test_data = np.array(
                                    test_data.copy().drop(columns=dropped_columns, errors='ignore'),
                                    dtype=np.float64)
                                test_dataset = TensorDataset(torch.tensor(np_test_data).float())
                                test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                                                             shuffle=False, num_workers=4)
                                predictor = L.Trainer(devices=1)
                                predictions_all_batches = predictor.predict(model, dataloaders=test_generator)
                                flow_labels = [1 if z >= 0 else 0 for z in
                                               [y.tolist() for x in predictions_all_batches for y in x]]
                                result["flow_query"] = flow_query
                                result["flow_labels"] = list(flow_labels)
                                result["flow_test_f1"] = precision_score(test_labels, flow_labels)
                                result["flow_time"]=time
                                with open("Results/animals/" +  str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name + ".json", "w") as f:
                                    json.dump(result, f)
                            else:
                                result["flow_query"] = no_boil_result["flow_query"]
                                result["flow_labels"] = no_boil_result["flow_labels"]
                                result["flow_test_f1"] = no_boil_result["flow_test_f1"]
                                result["flow_time"] = no_boil_result["flow_time"]
                                with open("Results/animals/"+ str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name + ".json", "w") as f:
                                    json.dump(result, f)

                        if "dt_flow_query" not in result:
                            print("-#-#-#DT")
                            no_boil_query = {}
                            if boil:
                                no_boil_result = {}
                                with open("Results/animals/" +  str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name.replace("_boiled", "") + ".json",
                                          "r") as f:
                                    no_boil_result = json.load(f)
                                no_boil_query = no_boil_result["dt_flow_query"]
                            tree_flow_query, model, boiled_effect, time = flow_reconstruction_lightning(train_data.copy(),
                                                                                                  set(ground_indices.copy()),
                                                                                                  copy.deepcopy(
                                                                                                      dt_query),
                                                                                                  dropped_columns.copy(),
                                                                                                  test_data.copy(),
                                                                                                  pos_weight,
                                                                                                  "animal_conv",
                                                                                                  boil,
                                                                                                  no_boil_query)
                            if (not boil) or (boil and boiled_effect):
                                np_test_data = np.array(
                                    test_data.copy().drop(columns=dropped_columns, errors='ignore'),
                                    dtype=np.float64)
                                test_dataset = TensorDataset(torch.tensor(np_test_data).float())
                                test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                                                             shuffle=False, num_workers=4)
                                predictor = L.Trainer(devices=1)
                                predictions_all_batches = predictor.predict(model, dataloaders=test_generator)
                                tree_flow_labels = [1 if z >= 0 else 0 for z in
                                                    [y.tolist() for x in predictions_all_batches for y in x]]
                                result["dt_flow_query"] = tree_flow_query
                                result["dt_flow_labels"] = list(tree_flow_labels)
                                result["dt_flow_test_f1"] = precision_score(test_labels, tree_flow_labels)
                                result["dt_time"]=time
                                with open("Results/animals/" + str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name + ".json", "w") as f:
                                    json.dump(result, f)
                            else:
                                result["dt_flow_query"] = no_boil_result["dt_flow_query"]
                                result["dt_flow_labels"] = no_boil_result["dt_flow_labels"]
                                result["dt_flow_test_f1"] = no_boil_result["dt_flow_test_f1"]
                                result["dt_time"]=no_boil_result["dt_time"]
                                with open("Results/animals/" + str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name + ".json", "w") as f:
                                    json.dump(result, f)
                        if "rule_flow_query" not in result:
                            print("-#-#-#RULES")
                            no_boil_query = {}
                            if boil:
                                no_boil_result = {}
                                with open("Results/animals/" + str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name.replace("_boiled", "") + ".json",
                                          "r") as f:
                                    no_boil_result = json.load(f)
                                no_boil_query = no_boil_result["rule_flow_query"]
                            rules_flow_query, model, boiled_effect, time = flow_reconstruction_lightning(
                                train_data.copy(), set(ground_indices.copy()), copy.deepcopy(rule_query),
                                dropped_columns.copy(), test_data.copy(), pos_weight, "animal_conv", boil,
                                no_boil_query)
                            if (not boil) or (boil and boiled_effect):
                                np_test_data = np.array(
                                    test_data.copy().drop(columns=dropped_columns, errors='ignore'),
                                    dtype=np.float64)
                                test_dataset = TensorDataset(torch.tensor(np_test_data).float())
                                test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                                                             shuffle=False, num_workers=4)
                                predictor = L.Trainer(devices=1)
                                predictions_all_batches = predictor.predict(model, dataloaders=test_generator)
                                rules_flow_labels = [1 if z >= 0 else 0 for z in
                                                     [y.tolist() for x in predictions_all_batches for y in x]]
                                result["rule_flow_query"] = rules_flow_query
                                result["rule_flow_labels"] = list(rules_flow_labels)
                                result["rule_flow_test_f1"] = precision_score(test_labels, rules_flow_labels)
                                result["rule_time"]=time
                                with open("Results/animals/" + str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name + ".json", "w") as f:
                                    json.dump(result, f)
                            else:
                                result["rule_flow_query"] = no_boil_result["rule_flow_query"]
                                result["rule_flow_labels"] = no_boil_result["rule_flow_labels"]
                                result["rule_flow_test_f1"] = no_boil_result["rule_flow_test_f1"]
                                result["rule_time"]=no_boil_result["rule_time"]
                                with open("Results/animals/" + str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name + ".json", "w") as f:
                                    json.dump(result, f)
                        if "nn_test_f1" not in result:
                            print("-#-#-#NN")
                            if boil:
                                no_boil_result = {}
                                with open("Results/animals/" + str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name.replace("_boiled", "") + ".json",
                                          "r") as f:
                                    no_boil_result = json.load(f)
                                result["nn_test_f1"] = no_boil_result["nn_test_f1"]
                                result["nn_labels"] = no_boil_result["nn_labels"]
                                with open("Results/animals/" + str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name + ".json", "w") as f:
                                    json.dump(result, f)
                            else:

                                # #How well does a single neural net perform
                                np_train_data = np.array(
                                    train_data.copy().drop(columns=dropped_columns, errors='ignore'),
                                    dtype=np.float64)
                                np_test_data = np.array(
                                    test_data.copy().drop(columns=dropped_columns, errors='ignore'),
                                    dtype=np.float64)
                                np_train_labels = np.array(train_labels, dtype=np.float64)
                                np_test_labels = np.array(test_labels, dtype=np.float64)
                                X_train, X_test, Y_train, Y_test = train_test_split(np_train_data, np_train_labels,
                                                                                    test_size=0.1, random_state=42)

                                model = res_cnn(pos_weight)  # Single_sequential(0.1)
                                model_history = trainModel_lightning(X_train, Y_train, X_test, Y_test, model=model,
                                                                     epochs=5)

                                # # print(model(torch.tensor(np_test_data).float()))
                                test_dataset = TensorDataset(torch.tensor(np_test_data).float())
                                test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                                                             shuffle=False, num_workers=4)

                                predictor = L.Trainer(devices=1)
                                predictions_all_batches = predictor.predict(model, dataloaders=test_generator)

                                nn_test_results = [1 if z[0] >= 0 else 0 for z in
                                                   [y.tolist() for x in predictions_all_batches for y in x]]
                                result["nn_test_f1"] = precision_score(np_test_labels, nn_test_results)
                                result["nn_labels"] = list(nn_test_results)
                                with open("Results/animals/"  + str(depth) + "_" + str(
                                        missing_var) + "_" + str(it) + result_name + ".json", "w") as f:
                                    json.dump(result, f)
