from threnpi.Data.dataloaders import load_housing_data, load_music_data, load_birds_data, load_objects_data, load_faces_data, load_medical_data
from threnpi.Problems.problem_generation.expert2prop import generate_random_theory
from threnpi.Problems.problem_generation.ilp2prop import rules2prop
from threnpi.Problems.problem_generation.decisiontree2prop import decisiontree2prop
from threnpi.src.propositional import clean_theory, AND, OR, VAR, NEURAL_PREDICATE, remove_variables
from sklearn.model_selection import train_test_split
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader
import weka.core.jvm as jvm
import os
import json
import copy
import random
import arff



def generate_problem(data, cat_columns, width, depth, missing_var, drop_cols=None):
    ## Generate a query in propositional logic where missing_var variables are deleted (sampled from drop_cols, the set of variables that are allowed).
    ## The resulting query should have a precision between 0.6 & 0.9.
    f1=1.0
    while f1>0.9 or f1<0.6:

        theory, counters, used=generate_random_theory(data, cat_columns, width, depth, [0,0,0])
        if drop_cols!=None:
            used=set(used).intersection(set(drop_cols))
        if len(used) > missing_var:
            for x in range(0,5):
                to_remove = list(random.sample(list(used), missing_var))
                removal=[]
                for i in theory.variables():
                    for r in to_remove:
                        if r in i.condition:
                            removal.append(i.name())
                final_theory = remove_variables(theory,set(removal))

                # Check if the final query is still good, otherwise generate another
                if final_theory!=None:
                    original=theory.get_covering_examples()
                    final=final_theory.get_covering_examples()
                    # there have to be at least 10% false positives
                    TP=len(original.intersection(final))
                    FP=len(final-original)
                    FN=len(original-final)
                    if len(original)>0.1*data.shape[0] and len(original)<data.shape[0] and len(final)>0:
                        f1=2*TP/(2*TP+FP+FN)
                    else:
                        f1=0
                else:
                    f1 = 0
                print("-->", f1, "/", to_remove)
                if 0.6<f1 and f1<0.9:
                    return theory, final_theory, set(to_remove), set(used) - set(to_remove)
    return theory, final_theory, set(to_remove), set(used)-set(to_remove)


 ###########################################################################
name="medical"
if True:
    full_data, categorical_columns, droppable_cols = load_medical_data()
    full_data = full_data[categorical_columns]
    # jvm.start(system_cp=True, packages=True)
    for it in range(0, 10):
        for width in range(2,3):
            for depth in range(2, 5):
                for missing_var in range(1, 5):
                    if False or not os.path.exists(
                            "Problems/"+name+"/expert_theories/"  + str(depth) + "_"+ str(width) + "_" + str(missing_var) + "_" + str(
                                    it) + "_expert.json"):
                        print(depth, width, missing_var, it)
                        correct = False
                        while not correct:
                            correct = True
                            original_theory, input_theory, dropped, used = generate_problem(full_data, categorical_columns,
                                                                                          width, depth, missing_var, droppable_cols)
                            dropped=list(dropped)
                            used=list(used)


                            ground_truth = full_data.iloc[list(original_theory.get_covering_examples())]
                            ground_indices=set(ground_truth.index)
                            full_labels = [int(x in ground_indices) for x in full_data.index]
                            train_data, validation_data, train_labels, validation_labels = train_test_split(full_data, full_labels,
                                                                                                test_size=0.1)

                            train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels,
                                                                                                test_size=0.1/0.9)
                            problem = {}
                            problem["train_indices"] = list(train_data.index)
                            problem["validation_indices"]=list(validation_data.index)
                            problem["test_indices"] = list(test_data.index)
                            problem["train_labels"] = train_labels
                            problem["validation_labels"]=validation_labels
                            problem["test_labels"] = test_labels
                            #
                            problem["original_query"] = original_theory.save()
                            problem["expert_query"] = input_theory.save()
                            problem["dropped_columns"] = dropped

                            with open("Problems/"+name+"/expert_theories/" + str(depth)+"_"+ str(width) + "_" + str(
                                    missing_var) + "_" + str(it) + "_expert.json", "w") as f:
                                json.dump(problem, f)

                            # # CONVERT TO WEKA FORMAT
                            # result = {}
                            #
                            # dropped_columns = problem["dropped_columns"]
                            # train_indices = problem["train_indices"]
                            # train_data = train_data[used]
                            # test_data = test_data[used]
                            #
                            # # generate labels for all of the data
                            # ground_truth = ground_truth[categorical_columns]
                            #
                            # train_data["Labels"] = [str(x) for x in train_labels]
                            # test_data["Labels"] = [str(x) for x in test_labels]
                            # # train_data=train_data.drop(columns=full_data.columns[14:], errors='ignore')
                            # # test_data=test_data.drop(columns=full_data.columns[14:], errors='ignore')
                            # attributes = [(j, 'NUMERIC') if train_data[j].dtypes in ['int64', 'float64'] else (
                            # j, train_data[j].unique().astype(str).tolist()) for j in train_data]
                            #
                            # arff_dic = {
                            #     'attributes': attributes,
                            #     'data': train_data.values,
                            #     'relation': 'myRel',
                            #     'names': train_data.columns,
                            #     'description': ''
                            # }
                            #
                            # with open("Data/animals/weka_data_discrete/"  + str(depth) + "_" + str(
                            #         missing_var) + "_" + str(it) + ".arff", "w", encoding="utf8") as f:
                            #     arff.dump(arff_dic, f)
                            #
                            # arff_dic = {
                            #     'attributes': attributes,
                            #     'data': test_data.values,
                            #     'relation': 'myRel',
                            #     'names': test_data.columns,
                            #     'description': ''
                            # }
                            #
                            # with open("Data/animals/weka_data_discrete/" + str(depth) + "_" + str(
                            #         missing_var) + "_" + str(it) + "_test.arff", "w", encoding="utf8") as f:
                            #     arff.dump(arff_dic, f)
                            #
                            # decision_tree = None
                            # decision_tree_f1 = None
                            # rules = None
                            # ruls_f1 = None
                            # for classifier_name, class_name, options in [
                            #     ("JRIP_", "weka.classifiers.rules.JRip", ["-F", "3", "-N", "2.0", "-O", "2", "-S", "1"]), (
                            #     "J48_", "weka.classifiers.trees.J48",
                            #     ["-C","0.25","-M","2"])]:
                            #     Cls = Classifier(classname=class_name, options=options)
                            #     loader = Loader(classname="weka.core.converters.ArffLoader")
                            #     training_data = loader.load_file(
                            #         "Data/animals/weka_data_discrete/" + str(depth) + "_" + str(
                            #             missing_var) + "_" + str(it) + ".arff")
                            #     training_data.class_is_last()
                            #     testing_data = loader.load_file(
                            #         "Data/animals/weka_data_discrete/"  + str(depth) + "_" + str(
                            #             missing_var) + "_" + str(it) + "_test.arff")
                            #     testing_data.class_is_last()
                            #     Cls.build_classifier(training_data)
                            #     evaluation = Evaluation(testing_data)
                            #     evaluation.test_model(Cls, testing_data)
                            #     if (evaluation.num_true_positives(1) + evaluation.num_false_positives(1)) > 0:
                            #         weka_f1 = 2*evaluation.num_true_positives(1) / (
                            #                     2*evaluation.num_true_positives(1) + evaluation.num_false_positives(1)+ evaluation.num_false_negatives(1))
                            #     else:
                            #         weka_f1 = 0
                            #     print(classifier_name, "-->", weka_f1)
                            #
                            #     if weka_f1 >0.9 or weka_f1<0.6:
                            #         correct = False
                            #         break;
                            #     else:
                            #         if classifier_name != "JRIP_":
                            #             dt_full = str(Cls)
                            #             # print(dt_full)
                            #             pos_rules=[x.strip() for x in str(Cls).split("\n")[2:-4] if len(x) > 0 and ": 0" not in x]
                            #             if len(pos_rules)>0:
                            #                 decision_tree, _, _ = decisiontree2prop(
                            #                     pos_rules, [0,0],
                            #                     {}, train_data)
                            #                 decision_tree = clean_theory(decision_tree)
                            #                 decision_tree_f1 = weka_f1
                            #             else:
                            #                 decision_tree=None
                            #                 decision_tree_f1=0
                            #         else:
                            #             rules_full = str(Cls)
                            #             rules = clean_theory(rules2prop(str(Cls), train_data))
                            #             if rules==None:
                            #                 print("Rules --> None")
                            #             rules_f1 = weka_f1
                            #     if decision_tree_f1 == rules_f1 or decision_tree_f1==0 or rules_f1==0 or (classifier_name != "JRIP_" and decision_tree==None) or (classifier_name == "JRIP_" and rules==None):
                            #         correct = False
                            #         break;
                        # print("#############################")
                        # print("Expert Knowledge")
                        # print(original_theory.save())
                        # print("--------------")
                        # print(input_theory.save())
                        # print("%%%%%%%%%%%%%%")
                        # print("DT")
                        # print(dt_full)
                        # print("--------------")
                        # print(decision_tree.save())
                        # print("%%%%%%%%%%%%%%")
                        # print("Rules")
                        # print(rules_full)
                        # print("--------------")
                        # print(rules.save())
                        # agreed=input("Is this conversion correct?")
                        # agreed="YES"
                        # if agreed=="YES":
                        #     problem["DT"] = decision_tree.save()
                        #     problem["DT_f1"] = decision_tree_f1
                        #     problem["Rules"] = rules.save()
                        #     problem["Rules_f1"] = rules_f1
                        #     problem["Rules_full"] = rules_full
                        #     problem["DT_full"] = dt_full
                        #     with open("Problems/animals/" + str(depth) + "_" + str(
                        #             missing_var) + "_" + str(it) + "_small.json", "w") as f:
                        #         json.dump(problem, f)


