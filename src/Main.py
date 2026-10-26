
from nether.Problems.example_problems import example_1,example_2
from nether.src.heuristic import get_ordered_operations, apply_operation
from nether.src.propositional import AND, OR, VAR, NEURAL_PREDICATE, remove_variables
from nether.src.construct_model import construct_model
from nether.src.train_model import train_model, validate_model
from sklearn.metrics import f1_score as f1_metric
import random
from copy import deepcopy
import time
import json
import math


def approximate_revision(data, theory,labels,cat_cols, edit_distance, k,mode, freeze, alpha=0.5, epochs=100, estimate="sharpe", device=[0,], symb_allowed=True, neural_allowed=True, test_data=[], test_labels=[]):
    """

    :param data: tuple containing training and validation data
    :param theory: propositional theory
    :param labels: labels matching with the data
    :param edit_distance: maximum amount of allowed edits
    :param k: amount of "training" runs used to initialize alpha, if k=0 and alpha is not given, alpha=0.5
    :param alpha: inital alpha value
    :param freeze: do we freeze previous changes
    :return:
    """
    random.seed(10)
    timing={"generating_modifications":[],"predicting_k":[],"constructing_model":[],"training_neural_net":[]}
    train_data, validation_data=data
    train_labels, validation_labels=labels
    pos_weight=(len(train_labels)-sum(train_labels))/len(train_labels)

    #setting the covering
    theory.set_covering_examples(validation_data, device=device)
    # --> find all possible neural predicate operations and select 'k' (only select the ones that can actually improve (otherwise there is only negative training data)
    before=time.time()
    operations, symb_ops = get_ordered_operations(theory, train_data,validation_data, train_labels,validation_labels,cat_cols,0.5, symb_allowed=symb_allowed, neural_allowed=neural_allowed, device=device)  # alpha doesn't matter here
    # print("Operations: ", operations)
    # print("Symbolic ops: ", symb_ops)
    timing["generating_modifications"].append(time.time()-before)
    no_op = [x for x in symb_ops if x[0] == "O"][0]
    #Estimate alpha by training 'k' dummy operations
    if k>0:
        before=time.time()
        print("#### ESTIMATING ALPHA ####")
        # print("Possible operations: ", operations)
        test_runs=random.sample([x for x in operations if x[3]>no_op[1]], min(len([x for x in operations if x[3]>no_op[1]]), k)) #only sample operations that can actually improve
        alpha_values=[]
        est_counter=1
        for operation in test_runs:
            print("--- ESTIMATION ", est_counter, "/", len(test_runs))
            # print("Perform operation: ", operation)
            est_counter+=1
            dummy_theory=deepcopy(theory)
            altered_theory=apply_operation(dummy_theory, operation, mode,device=device)
            # print("Altered theory: ", altered_theory.show())
            model=construct_model(altered_theory, train_data.columns, pos_weight, mode, freeze)
            train_model(model, (train_data, validation_data), (train_labels, validation_labels), epochs, True, device=device)
            f1_score=validate_model(model, validation_data, validation_labels, device=device)
            alpha_values.append((f1_score-operation[2])/(operation[3]-operation[2]))
            # print("Alpha --> ", alpha_values[-1])
        if len(alpha_values)>0:
            alpha=sum(alpha_values)/len(alpha_values)
            # print(alpha_values, "-->", alpha)
        else:
            alpha=0.5
        timing["predicting_k"].append(time.time()-before)
    result=[no_op[1],None, theory] #f1_score, model, theory
    print("#### START REVISION ###")
    previous_op=[]
    edit_counter=0

    operations_path=[]
    covered_test_indices = result[2].get_covering_examples(test_data, device=device)
    path_prediction = [x in covered_test_indices for x in test_data.index]
    init_f1_score=f1_metric(test_labels, path_prediction)
    operations_path.append(["T",init_f1_score,no_op[1], alpha])
    #stops when too much edits
    while edit_counter<edit_distance:

        print("---- SEARCHING FOR OPERATIONS ----")
        before=time.time()
        operations, symb_operations = get_ordered_operations(result[2], train_data, validation_data, train_labels,
                                                      validation_labels, cat_cols, alpha=alpha, symb_allowed=symb_allowed, estimate=estimate, neural_allowed=neural_allowed, device=device)  # alpha doesn't matter here

        timing["generating_modifications"].append(time.time() - before)
        # print("----- REM OPERATIONS -----")
        # for i in symb_operations:
        #     print(i)
        # print("---- ADDING OPERATIONS ----")
        # for i in operations:
        #     print(i)
        operations=[x for x in operations if x[1]>0 and x[0] not in [z[0] for z in previous_op]] #all_operations that are not done yet and perform better than the risk free
        if len(operations)==0:
            operation=[x for x in symb_operations if x not in previous_op][0] #Take the best risk-free operation
        else:
            operation=operations[0]
        if operation[0]=="O":
            # print("BEST OPERATION IS NEUTRAL")
            return result, timing, operations_path
        previous_op.append(operation)
        print("--> APPLYING : ", operation)
        dummy_theory = deepcopy(result[2])
        altered_theory = apply_operation(dummy_theory, operation, mode,device=device)
        if len(operation)>2 and "S" not in operation[0]: #NP operation
            before=time.time()
            model = construct_model(altered_theory, train_data.columns, pos_weight, mode, freeze)
            timing["constructing_model"].append(time.time()-before)
            print("---- TRAINING ----")
            before=time.time()
            train_model(model, (train_data, validation_data), (train_labels, validation_labels), epochs, True, device=device)
            timing["training_neural_net"].append(time.time()-before)
            print("---- VALIDATION ----")

            f1_score = validate_model(model, validation_data, validation_labels, device=device)
            if estimate=="sharpe":
                print("---- UPDATA ALPHA ----")
                print("Previous: ", alpha)
                new_alpha=(f1_score-operation[2])/(operation[3]-operation[2])
                print("Update: ", new_alpha)
                alpha=(alpha+new_alpha)/2
                print("--> ", alpha)

            if f1_score>result[0]:
                result = [f1_score, model, altered_theory]
                for np in altered_theory.neural_predicates():
                    # update the neural predicates with their trained model
                    np.neural_layer = result[1].neural_layers[np.id]

                covered_test_indices = result[2].get_covering_examples(test_data, device=device)
                path_prediction = [x in covered_test_indices for x in test_data.index]
                path_f1_score = f1_metric(test_labels,path_prediction)
                operations_path.append([operation[0], path_f1_score, result[0], alpha])
                edit_counter+=1

        else: #Symbolic revision
            if operation[1]>result[0]:
                edit_counter += 1
                if len(altered_theory.neural_predicates())>0: #a previous operation introduced a neural predicate, so there is a lightning model
                    before = time.time()
                    model = construct_model(altered_theory, train_data.columns, pos_weight, mode, freeze)
                    timing["constructing_model"].append(time.time() - before)
                    print("---- VALIDATION ----")
                    f1_score = validate_model(model, validation_data, validation_labels, device=device)
                    if f1_score > result[0]:
                        result = [f1_score, model, altered_theory]
                        covered_test_indices = result[2].get_covering_examples(test_data, device=device)
                        path_prediction = [x in covered_test_indices for x in test_data.index]
                        path_f1_score = f1_metric(test_labels, path_prediction)
                        operations_path.append([operation[0], path_f1_score, result[0], alpha])
                else: #there are no neural predicates, thus inference can be done just on the altered theory (hence the model is None)
                    result=[operation[1], None, altered_theory]
                    covered_test_indices = result[2].get_covering_examples(test_data, device=device)
                    path_prediction = [x in covered_test_indices for x in test_data.index]
                    path_f1_score = f1_metric(test_labels, path_prediction)
                    operations_path.append([operation[0], path_f1_score, result[0], alpha])
    return result, timing, operations_path
