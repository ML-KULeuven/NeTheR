from operator import itemgetter

import math
import copy

from nether.src.DT import learn_decision_tree
from nether.src.propositional import AND, OR, VAR, clean_theory, load_theory, NEURAL_PREDICATE, remove_variables
from sklearn.metrics import f1_score


def apply_operation(theory, operation, mode, device=[0,]):
    op_name=operation[0]
    if op_name[0]=="R":
        # print("Removing variables from: ", theory.save(), operation)
        return remove_variables(theory, [op_name[1:]], device=device)
    elif op_name[0]=="O":
        return theory
    elif op_name[0]=="S":
        # print("ADDING SYMBOLIC: ", op_name, operation[2])
        new_var=VAR(max([-1]+[x.id for x in theory.variables()])+1, operation[2])
        if op_name[1]=="A": #Add
            if op_name[2]=="D":
                return OR(max([-1] + [x.id for x in theory.nodes() if x.is_or()]) + 1, [theory, new_var],device=device)
            elif op_name[2]=="C":
                return AND(max([-1] + [x.id for x in theory.nodes() if x.is_and()]) + 1, [theory, new_var],device=device)
        else:
            node = [x for x in theory.nodes() if x.name() == op_name[1:]][0]
            if node.is_or() or node.is_and():
                node.children.append(new_var)
            else:
                parent = [x for x in theory.nodes() if (x.is_or() or x.is_and()) and node in x.children][0]
                parent.children.remove(node)
                if parent.is_or():
                    parent.children.append(
                        AND(max([-1] + [x.id for x in theory.nodes() if x.is_and()]) + 1, [node, new_var],device=device))
                else:
                    parent.children.append(OR(max([-1] + [x.id for x in theory.nodes() if x.is_or()]) + 1, [node, new_var],device=device))
            return theory

    else:
        np=NEURAL_PREDICATE(max([-1]+[x.id for x in theory.neural_predicates()])+1, mode)
        if op_name[0]=="A":
            node = [x for x in theory.nodes() if x.name() == op_name[2:]][0]
            if op_name[1]=="D":
                return OR(max([-1] + [x.id for x in theory.nodes() if x.is_or()]) + 1, [node, np],device=device)
            else:
                return AND(max([-1] + [x.id for x in theory.nodes() if x.is_and()]) + 1, [node, np],device=device)
        else:
            node=[x for x in theory.nodes() if x.name()==op_name][0]
            if node.is_or() or node.is_and():
                node.children.append(np)
            else:
                parent=[x for x in theory.nodes() if (x.is_or() or x.is_and()) and node in x.children][0]
                parent.children.remove(node)
                if parent.is_or():
                    parent.children.append(AND(max([-1]+[x.id for x in theory.nodes() if x.is_and()])+1, [node, np],device=device))
                else:
                    parent.children.append(OR(max([-1]+[x.id for x in theory.nodes() if x .is_or()]) + 1, [node, np],device=device))
            return theory


def adding_np_is_allowed(theory, sub):
    if sub.is_and():
        if len([x for x in sub.children if x.is_neural_predicate()])>0:
            return False
        else:
            return True
    elif sub.is_or():
        if len([x for x in sub.children if x.is_neural_predicate()])>0:
            return False
        else:
            return True

def symbolic_revision(theory, node, covered_train_data,  val_data, cat_cols, add_in_conj=False, device=[0,]):
    # print("Symbolic revision on ", theory.save())
    # print("Node: ", node.save())
    # print("Train data: ", covered_train_data)
    train_data = covered_train_data.loc[list(node.get_covering_examples(covered_train_data, [node.name(), covered_train_data.index, add_in_conj], device=device))]
    # print("Train data: \n", train_data[cat_cols+["label"]])
    dt, classifier = learn_decision_tree(train_data, [int(x) for x in train_data["label"]], cat_cols, single_split=True)
    # print("Symbolic revision: ", node.name(), "DT: ", dt)
    result=[]
    if dt==None:
        # print("No DT Learned")
        return result
    # else:
        # print("Learned dt: ", dt.condition)
    operator_name = "S"
    new_theory = copy.deepcopy(theory)
    if theory==node or node.is_var() or node.is_neural_predicate():
        result=[]
        new_theory = copy.deepcopy(theory)
        if  (not add_in_conj) and (not node.is_or()):
            if theory == node:
                operator_name = "SAD"
            new_theory= apply_operation(new_theory, [operator_name+node.name(),0,dt.condition],"",device=device)

            result.append((operator_name + node.name(), f1_score(val_data["label"],
                                                [x in new_theory.get_covering_examples(val_data, device=device) for x in
                                                 val_data.index]), dt.condition))
        elif add_in_conj and (not node.is_and()):
            if theory == node:
                operator_name = "SAC"
            new_theory= apply_operation(new_theory, [operator_name+node.name(),0,dt.condition],"",device=device)
            result.append((operator_name + node.name(), f1_score(val_data["label"],
                                                [x in new_theory.get_covering_examples(val_data, device=device) for x in
                                                 val_data.index]), dt.condition))
    if (add_in_conj and node.is_and()) or (not add_in_conj and node.is_or()):
        operator_name = "S"
        new_theory = apply_operation(new_theory, [operator_name+node.name(),0,dt.condition], "",device=device)
        result.append((operator_name + node.name(), f1_score(val_data["label"], [x in new_theory.get_covering_examples(val_data, device=device) for x in val_data.index]), dt.condition))
    return result

def get_add_conj_operations(theory, train_data, train_labels, val_data, val_labels, cat_cols, symb_allowed, neural_allowed,device=[0,]):
    #Find all conjunctions and add a neural predicate
    #Return the F1-scores of the best case (NP does a perfect job, only removes FP) and worst case (NP does a terrible job, only removes TP)

    operations=[]
    symb_operations=[]
    train_data["label"]=train_labels
    val_data["label"]=val_labels

    val_indices=theory.get_covering_examples(val_data, device=device)
    covered_val_data=val_data.loc[list(val_indices)]
    train_indices = theory.get_covering_examples(train_data, device=device)
    covered_train_data = train_data.loc[list(train_indices)]

    positives=set(covered_val_data.index[covered_val_data['label'] == 1].tolist())
    negatives=val_indices-positives
    theory.set_circuit_flow(val_indices)
    conjunctions=[node for node in theory.nodes() if node.is_and()]

    #CONSTRAINTS PREDICATES TO BE ADDED
    for conj in conjunctions:
        #perform symbolic revision: in a conjunction, it has to learn that the false positives should be negatives, without hurting the positives
        if symb_allowed:
            symb_operations+=symbolic_revision(theory, conj, covered_train_data, val_data, cat_cols, add_in_conj=True, device=device)

        if adding_np_is_allowed(theory, conj) and neural_allowed:

            #alter the init_flows such that worst and best case are put through the circuit
            worst_case=theory.get_covering_examples(covered_val_data, [conj.name(),negatives], device=device)
            best_case=theory.get_covering_examples(covered_val_data, [conj.name(),positives], device=device)
            worst_labels=[x in worst_case for x in val_data.index]
            best_labels = [x in best_case for x in val_data.index]
            worst=f1_score(val_labels, worst_labels)
            best=f1_score(val_labels, best_labels)
            operations.append((conj.name(), worst, best))
    # If the theory is a single var or a disjunction, add a conjunction on top
    if theory.is_var() or theory.is_or():
        if symb_allowed:
            symb_operations+=symbolic_revision(theory, theory, covered_train_data, val_data, cat_cols, add_in_conj=True, device=device)
        if neural_allowed:
            worst_case=theory.get_covering_examples(val_data, device=device).intersection(negatives)
            best_case = theory.get_covering_examples(val_data, device=device).intersection(positives)
            worst_labels = [x in worst_case for x in val_data.index]
            best_labels = [x in best_case for x in val_data.index]
            worst = f1_score(val_labels, worst_labels)
            best = f1_score(val_labels, best_labels)
            operations.append(("AC"+theory.name(), worst, best))

    #find all vars that are a child of a disjunction
    disjunctions = [node for node in theory.nodes() if node.is_or()]

    for i in disjunctions:
        for child in i.children:
            if symb_allowed and (child.is_var() or child.is_neural_predicate()):
                symb_operations += symbolic_revision(theory, child, covered_train_data, val_data,cat_cols, add_in_conj=True, device=device)
            if child.is_var() and neural_allowed:
                worst_case = theory.get_covering_examples(val_data, [child.name(), negatives, True], device=device)
                best_case = theory.get_covering_examples(val_data, [child.name(), positives, True], device=device)

                worst_labels = [x in worst_case for x in val_data.index]
                best_labels = [x in best_case for x in val_data.index]
                worst = f1_score(val_labels, worst_labels)
                best = f1_score(val_labels, best_labels)
                operations.append((child.name(), worst, best))
    return operations, symb_operations

def get_add_disj_operations(theory, train_data, train_labels, val_data, val_labels, cat_cols, symb_allowed, neural_allowed, device=[0,]):

    operations=[]
    symb_operations=[]
    train_data["label"]=train_labels
    val_data["label"]=val_labels

    covered_val_indices = theory.get_covering_examples(val_data, device=device)
    not_covered_val_indices=set(val_data.index)-covered_val_indices
    other_data=val_data.loc[list(not_covered_val_indices)]
    positives=set(other_data.index[other_data['label'] == 1].tolist())
    negatives=not_covered_val_indices-positives

    covered_train_indices = theory.get_covering_examples(train_data, device=device)
    not_covered_train_indices = set(train_data.index) - covered_train_indices
    not_covered_training_data = train_data.loc[list(not_covered_train_indices)]

    #Only select disjunctions that don't have child or parent disjunction that already contain a neural predicate
    disjunctions = [node for node in theory.nodes() if node.is_or()]
    allowed_disjunctions=[]
    for disj in disjunctions:
        if symb_allowed and len(not_covered_training_data)>0:
            symb_operations += symbolic_revision(theory, disj, not_covered_training_data, val_data, cat_cols, device=device)
        if adding_np_is_allowed(theory, disj) and neural_allowed:

            #alter the init_flows such that worst and best case are put through the circuit
            worst_case = theory.get_covering_examples(val_data, [disj.name(), negatives], device=device)
            best_case = theory.get_covering_examples(val_data, [disj.name(), positives], device=device)

            worst_labels=[x in worst_case for x in val_data.index]
            best_labels = [x in best_case for x in val_data.index]
            worst=f1_score(val_labels, worst_labels)
            best=f1_score(val_labels, best_labels)
            operations.append((disj.name(), worst, best))

    #If the theory is a single var or a conjunction, add a disjunction on top
    if theory.is_var() or theory.is_and():
        if symb_allowed and len(not_covered_training_data)>0:
            symb_operations += symbolic_revision(theory, theory, not_covered_training_data, val_data, cat_cols, device=device)
        if neural_allowed:
            worst_case=theory.get_covering_examples(val_data, device=device).union(negatives)
            best_case = theory.get_covering_examples(val_data, device=device).union(positives)
            worst_labels = [x in worst_case for x in val_data.index]
            best_labels = [x in best_case for x in val_data.index]
            worst = f1_score(val_labels, worst_labels)
            best = f1_score(val_labels, best_labels)
            operations.append(("AD"+theory.name(), worst, best))


    #For all VARS in conjunction, add a neural predicate in disjunction
    conjunctions = [node for node in theory.nodes() if node.is_and()]

    for i in conjunctions:
        for child in i.children:
            if symb_allowed and (child.is_var() or child.is_neural_predicate()) and len(not_covered_training_data)>0:
                symb_operations += symbolic_revision(theory, child, not_covered_training_data, val_data,cat_cols, device=device)
            if child.is_var() and neural_allowed:# and adding_np_is_allowed(theory, child):
                worst_case = theory.get_covering_examples(val_data, [child.name(), negatives, False], device=device)
                best_case = theory.get_covering_examples(val_data, [child.name(), positives, False], device=device)
                worst_labels = [x in worst_case for x in val_data.index]
                best_labels = [x in best_case for x in val_data.index]
                worst = f1_score(val_labels, worst_labels)
                best = f1_score(val_labels, best_labels)
                operations.append((child.name(), worst, best))
    return operations, symb_operations

def get_ordered_operations(theory, train_data,val_data, train_labels,val_labels,cat_cols, alpha=.75, estimate="sharpe", symb_allowed=True, neural_allowed=True, device=[0,]):
    # print("Ordered: ", device)
    theory.set_covering_examples(val_data, device=device)
    all_operations=[]
    symb_operations=get_removal_operations(theory, val_data, val_labels, symb_allowed, device=device)
    conj_np, conj_symb=get_add_conj_operations(theory, train_data, train_labels, val_data, val_labels, cat_cols, symb_allowed, neural_allowed, device=device)
    disj_np, disj_symb=get_add_disj_operations(theory, train_data, train_labels, val_data, val_labels, cat_cols, symb_allowed, neural_allowed, device=device)

    symb_operations+=conj_symb+disj_symb
    symb_operations=sorted(symb_operations, key=lambda x: x[1], reverse=True)
    O_r = symb_operations[0]  # Used in the Sharpe ratio to score the neural predicates
    # print("Symbolic operators: ", symb_operations)
    # print("Best exact operation: ", O_r)
    if estimate=="sharpe" or estimate=="mean":
        alpha=min(0.99,alpha)
        for op in conj_np+disj_np:
            name=op[0]
            W=op[1]
            # W=no_op
            B=op[2]
            if W==B:
                sharpe=-1
            else:
                try:
                    sharpe=(alpha*B+(1-alpha)*W-O_r[1])/math.sqrt(alpha*((1-alpha)*B-(1-alpha)*W)**2+(1-alpha)*(alpha*W-alpha*B)**2)
                except:
                    sharpe=-1
            all_operations.append((name, sharpe, W, B))
    elif estimate=="sortino":
        alpha=min(0.99,alpha)
        for op in conj_np+disj_np:
            name=op[0]
            W=op[1]
            # W=no_op
            B=op[2]
            # print("Sortino --> ", "B: ", B, "W: ", W, "O_r:", O_r[1])
            if W>=O_r[1]:
                # print("W>O_r")
                sortino=-1
            else:
                try:
                    sortino=(alpha*B+(1-alpha)*W-O_r[1])/math.sqrt((1-alpha)*(W-O_r[1])**2)
                except Exception as e:
                    # print("Error: ",e)
                    sortino=-1
            # print("--> sortino: ", sortino)
            all_operations.append((name, sortino, W, B))
    elif estimate=="max":
        for op in conj_np+disj_np:
            name=op[0]
            B=op[2]
            all_operations.append((name, B-O_r[1], B))
    elif estimate == "min":
        for op in conj_np + disj_np:
            name = op[0]
            W = op[1]
            all_operations.append((name, W - O_r[1], W))
    all_operations.sort(key=lambda x:x[1], reverse=True)

    return all_operations, symb_operations

def get_removal_operations(theory, val_data, val_labels, symb_allowed, device=[0,]):
    # print("GET REMOVAL: ", len(theory.get_covering_examples()))
    all_operations=[("O",f1_score(val_labels, [x in theory.get_covering_examples(device=device) for x in val_data.index]))]
    # print("ALL OPERATIONS: ", all_operations)
    all_vars = theory.variables() + theory.neural_predicates()
    if len(all_vars)>1 and symb_allowed:
        for var in all_vars:
            new_theory=remove_variables(theory, [var.name()],device=device)
            # print("REMOVED ", var, len(new_theory.get_covering_examples()))
            # new_theory.get_covering_examples(data)
            # print(len(new_theory.variables()))
            all_operations.append(("R"+var.name(), f1_score(val_labels, [x in new_theory.get_covering_examples(device=device) for x in val_data.index])))
    return all_operations




