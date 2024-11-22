from sympy.logic.inference import valid

from threnpi.Problems.example_problems import example_1,example_2
from threnpi.src.heuristic import get_ordered_operations
from threnpi.src.propositional import AND, OR, VAR, NEURAL_PREDICATE, remove_variables
from threnpi.src.construct_model import construct_model
from threnpi.src.train_model import train_model, validate_model
import random
from copy import deepcopy

import json
import math


def apply_operation(theory, operation, mode):
    print(operation)
    op_name=operation[0]
    if op_name[0]=="R":
        return remove_variables(theory, [op_name[1:]])
    elif op_name[0]=="O":
        return theory
    else:
        np=NEURAL_PREDICATE(max([-1]+[x.id for x in theory.neural_predicates()])+1, mode)
        if op_name[0]=="A":
            node = [x for x in theory.nodes() if x.name() == op_name[2:]][0]
            if op_name[1]=="D":
                return OR(max([-1] + [x.id for x in theory.nodes() if x.is_or()]) + 1, [node, np])
            else:
                return AND(max([-1] + [x.id for x in theory.nodes() if x.is_and()]) + 1, [node, np])
        else:
            node=[x for x in theory.nodes() if x.name()==op_name][0]
            if node.is_or() or node.is_and():
                node.children.append(np)
            else:
                parent=[x for x in theory.nodes() if (x.is_or() or x.is_and()) and node in x.children][0]
                parent.children.remove(node)
                if parent.is_or():
                    parent.children.append(AND(max([-1]+[x.id for x in theory.nodes() if x.is_and()])+1, [node, np]))
                else:
                    parent.children.append(OR(max([-1]+[x.id for x in theory.nodes() if x .is_or()]) + 1, [node, np]))
            return theory

def approximate_revision(data, theory,labels, edit_distance, k,h,p,merge,mode, freeze, alpha=0.5, epochs=100, estimate="sharpe"):
    """

    :param data: tuple containing training and validation data
    :param theory: propositional theory
    :param labels: labels matching with the data
    :param edit_distance: maximum amount of allowed edits
    :param k: amount of "training" runs used to initialize alpha, if k=0 and alpha is not given, alpha=0.5
    :param h: amount of hill_climbing
    :param p: patience value --> if the best operation is validated lower than the previous one for p times, stop
    :param merge: do we allow multiple neural predicates in a path
    :param alpha: inital alpha value
    :param freeze: do we freeze previous changes
    :return:
    """
    random.seed(10)
    train_data, validation_data=data
    train_labels, validation_labels=labels
    pos_weight=(len(train_labels)-sum(train_labels))/len(train_labels)
    init_number_variables=len(theory.variables())

    #setting the covering


    #Estimate alpha by training 'k' dummy operations
    if k>0:
        print("#### ESTIMATING ALPHA ####")
        theory.set_covering_examples(validation_data)
        # --> find all possible operations and select 'k' (only select the ones that can actually improve (otherwise there is only negative training data)
        operations, rem_ops = get_ordered_operations(theory, validation_data, validation_labels, 0.5) #alpha doesn't matter here
        no_op=[x for x in rem_ops if x[0]=="O"][0]
        print("Possible operations: ", operations)
        test_runs=random.sample([x for x in operations if x[3]>no_op[1]], min(len([x for x in operations if x[3]>no_op[1]]), k)) #only sample operations that can actually improve
        alpha_values=[]
        est_counter=1
        for operation in test_runs:
            print("--- ESTIMATION ", est_counter, "/", len(test_runs))
            est_counter+=1
            dummy_theory=deepcopy(theory)
            altered_theory=apply_operation(dummy_theory, operation, mode)
            model=construct_model(altered_theory, train_data.columns, pos_weight, mode, freeze)
            train_model(model, (train_data, validation_data), (train_labels, validation_labels), epochs, True)
            f1_score=validate_model(model, validation_data, validation_labels)
            alpha_values.append((f1_score-operation[2])/(operation[3]-operation[2]))
            print("Alpha --> ", alpha_values[-1])
        if len(alpha_values)>0:
            alpha=sum(alpha_values)/len(alpha_values)
            print(alpha_values, "-->", alpha)
        else:
            alpha=0.5
    global_best_result=[0,None, theory]
    print("#### START REVISION ###")
    previous_best_f1=0
    patience_counter=0
    previous_op=[]
    #stops when too much edits (#neural predicates + #removed logical variables) or when patience gets too low
    while len(theory.neural_predicates())+init_number_variables-len(theory.variables())<edit_distance or patience_counter>=p:
        print("---- SEARCHING FOR OPERATIONS ----")
        operations, rem_operations=get_ordered_operations(theory, validation_data,validation_labels, alpha, estimate=estimate)
        print("----- REM OPERATIONS -----")
        for i in rem_operations:
            print(i)
        print("---- ADDING OPERATIONS ----")
        for i in operations:
            print(i)
        operations=[x for x in operations if x[1]>0 and x[0] not in previous_op][:h] #if h=1, hill climbing is greedy
        if len(operations)<h:
            operations.append(rem_operations[0])

        best_result=[0,None,theory]
        best_operation=None
        for operation in operations:
            print("--> APPLYING : ", operation)
            dummy_theory = deepcopy(theory)
            altered_theory = apply_operation(dummy_theory, operation, mode)
            if len(operation)>2:
                model = construct_model(altered_theory, train_data.columns, pos_weight, mode, freeze)
                print("---- TRAINING ----")
                train_model(model, (train_data, validation_data), (train_labels, validation_labels), epochs, True)
                print("---- VALIDATION ----")

                f1_score = validate_model(model, validation_data, validation_labels)
                if f1_score>best_result[0]:
                    best_result=[f1_score, model, altered_theory]
                    best_operation=operation
            else:
                if operation[1]>best_result[0]:
                    if len(altered_theory.neural_predicates())>0: #a previous operation introduced a neural predicate, so there is a lightning model
                        model = construct_model(altered_theory, train_data.columns, pos_weight, mode, freeze)
                        print("---- VALIDATION ----")
                        f1_score = validate_model(model, validation_data, validation_labels)
                        if f1_score > best_result[0]:
                            best_result = [f1_score, model, altered_theory]
                            best_operation = operation
                    else: #there are no neural predicates, thus inference can be done just on the altered theory (hence the model is None)
                        best_result=[operation[1], None, altered_theory]
                        best_operation=operation
        if best_operation!=None:
            previous_op.append(best_operation[0])
        else:
            previous_op.append(operation[0])
        theory=best_result[2]
        #update alpha by taking the mean of current alpha and the new result
        if best_operation!= None and len(best_operation)>2: #Not a removal operation
            if estimate=="sharpe":
                print("---- UPDATA ALPHA ----")
                print("Previous: ", alpha)
                new_alpha=(f1_score-best_operation[2])/(best_operation[3]-best_operation[2])
                print("Update: ", new_alpha)
                alpha=(alpha+new_alpha)/2
                print("--> ", alpha)
            for np in theory.neural_predicates():
                #update the neural predicates with their trained model
                np.neural_layer=best_result[1].neural_layers[np.id]

        #Bookkeeping patience
        if best_result[0]<=previous_best_f1:
            patience_counter+=1
        else:
            patience_counter=0
        previous_best_f1=best_result[0]
        #Storing best model (used through patience)
        if best_result[0]>global_best_result[0]:
            global_best_result=best_result
        if best_operation!=None and best_operation[0]=="O":
            return global_best_result
    return global_best_result
#
#
# full_data, cat_cols, drop_cols=load_small_animal_data()
# original_theory, expert_theory=example_2()
#
#
# original_indices=original_theory.get_covering_examples(full_data)
# training_indices=expert_theory.get_covering_examples(full_data)
# TP=len(original_indices.intersection(training_indices))
# FP=len(training_indices-original_indices)
# FN=len(original_indices-training_indices)
# print("F1=", 2*TP/(2*TP+FP+FN), "TP=", TP, "FP=", FP, "FN=", FN)
#
# all_indices=set(full_data.index)
# training_indices=set(random.sample(list(all_indices), int(math.ceil(0.9*len(all_indices)))))
# validation_indices=all_indices-training_indices
# training_data=full_data.loc[list(training_indices)].copy()
# validation_data=full_data.loc[list(validation_indices)].copy()
#
#
#
# original_training_indices=original_theory.get_covering_examples(training_data)
# original_validation_indices=original_theory.get_covering_examples(validation_data)
# training_labels=[x in original_training_indices for x in set(training_data.index)]
# validation_labels=[x in original_validation_indices for x in set(validation_data.index)]
#
# data=(training_data, validation_data)
# labels=(training_labels, validation_labels)
#
#
# edit_distance=5
# k=0
# alpha=0.97
# h=1
# p=5
# merge=False
# freeze=True
# mode="animal_conv"
# best_result=approximate_revision(data, expert_theory,labels, edit_distance, k,h,p,merge, mode,freeze, alpha)
# print("DONE")
# print("MODEL: ", best_result[1])
# print("F1-score", best_result[0])
# print("Theory: ")
# print(best_result[2].show())