from operator import itemgetter
import json
from PIL.ImageOps import cover
from sympy.polys.polyconfig import query
import math
from threnpi.src.propositional import AND, OR, VAR, clean_theory, load_theory, NEURAL_PREDICATE, remove_variables
from sklearn.metrics import f1_score


def adding_np_is_allowed(theory, sub):
    if sub.is_and():
        if len([x for x in sub.children if x.is_neural_predicate()])>0:
            return False
        else:
            return True
        # all_conj_children = [n for n in sub.nodes() if "AND" in str(type(n))]
        # allowed = True
        # for node in all_conj_children:
        #     for child in node.children:
        #         if "NEURAL_PREDICATE" in str(type(child)):
        #             allowed = False
        # if allowed:
        #     parent = [node for node in theory.nodes() if
        #               ("OR" in str(type(node)) or "AND" in str(type(node))) and sub in node.children]
        #     while len(parent) > 0:
        #         parent = parent[0]
        #         if "AND" in str(type(parent)) and len([x for x in parent.children if "NEURAL_PREDICATE" in str(type(x))]) > 0:
        #             allowed = False
        #             break;
        #         parent = [node for node in theory.nodes() if
        #                   ("OR" in str(type(node)) or "AND" in str(type(node))) and parent in node.children]
        # if allowed:
        #     return True
        # else:
        #     return False
    elif sub.is_or():
        if len([x for x in sub.children if x.is_neural_predicate()])>0:
            return False
        else:
            return True
        # all_disj_children = [n for n in sub.nodes() if "OR" in str(type(n))]
        # allowed = True
        # for node in all_disj_children:
        #     for child in node.children:
        #         if "NEURAL_PREDICATE" in str(type(child)):
        #             allowed = False
        # if allowed:
        #
        #     parent = [node for node in theory.nodes() if
    #                   ("OR" in str(type(node)) or "AND" in str(type(node))) and sub in node.children]
    #         while len(parent) > 0:
    #             parent = parent.pop(0)
    #             if "OR" in str(type(parent)) and len([x for x in parent.children if "NEURAL_PREDICATE" in str(type(x))]) > 0:
    #                 allowed = False
    #                 break;
    #             parent = [node for node in theory.nodes() if
    #                       ("AND" in str(type(node)) or "OR" in str(type(node))) and parent in node.children]
    #     if allowed:
    #         return True
    #     else:
    #         return False
    # elif "VAR" in str(type(sub)):
    #     allowed=True
    #     if allowed:
    #         parent = [node for node in theory.nodes() if
    #                   ("AND" in str(type(node)) or "OR" in str(type(node))) and sub in node.children]
    #         if "OR" in str(type(parent[0])):
    #             adding_type="AND"
    #         else:
    #             adding_type="OR"
    #
    #         while len(parent) > 0:
    #             parent = parent.pop(0)
    #             if adding_type in str(type(parent)) and len([x for x in parent.children if "NEURAL_PREDICATE" in str(type(x))]) > 0:
    #                 allowed = False
    #                 break;
    #             parent = [node for node in theory.nodes() if
    #                       ("AND" in str(type(node)) or "OR" in str(type(node))) and parent in node.children]
    #     if allowed:
    #         return True
    #     else:
    #         return False

def get_add_np_conj_operations(theory, data, labels):
    #Find all conjunctions and add a neural predicate
    #Return the F1-scores of the best case (NP does a perfect job, only removes FP) and worst case (NP does a terrible job, only removes TP)

    operations=[]
    data["label"]=labels
    print("Checking neural preds: ")
    for i in theory.neural_predicates():
        print(i.mode)
    original_indices=theory.get_covering_examples(data)
    original_data=data.loc[list(original_indices)]


    positives=set(original_data.query("label==1").index)
    negatives=original_indices-positives
    theory.set_circuit_flow(original_indices)
    conjunctions=[node for node in theory.nodes() if node.is_and()]


    #CONSTRAINTS PREDICATES TO BE ADDED
    allowed_conjunctions = []
    for conj in conjunctions:
        if adding_np_is_allowed(theory, conj):
            allowed_conjunctions.append(conj)

    for i in allowed_conjunctions:
        #alter the init_flows such that worst and best case are put through the circuit
        worst_case=theory.get_covering_examples(original_data, [i.name(),negatives])
        best_case=theory.get_covering_examples(original_data, [i.name(),positives])
        worst_labels=[x in worst_case for x in data.index]
        best_labels = [x in best_case for x in data.index]
        worst=f1_score(labels, worst_labels)
        best=f1_score(labels, best_labels)
        operations.append((i.name(), worst, best))
    # If the theory is a single var or a disjunction, add a conjunction on top
    if theory.is_var() or theory.is_or():
        worst_case=theory.get_covering_examples(data).intersection(negatives)
        best_case = theory.get_covering_examples(data).intersection(positives)
        worst_labels = [x in worst_case for x in data.index]
        best_labels = [x in best_case for x in data.index]
        worst = f1_score(labels, worst_labels)
        best = f1_score(labels, best_labels)
        operations.append(("AC"+theory.name(), worst, best))

    #find all vars that are a child of a disjunction
    disjunctions = [node for node in theory.nodes() if node.is_or()]

    for i in disjunctions:
        for child in i.children:
            if child.is_var():#  and adding_np_is_allowed(theory, child):
                worst_case = theory.get_covering_examples(data, [child.name(), negatives, True])
                best_case = theory.get_covering_examples(data, [child.name(), positives, True])

                worst_labels = [x in worst_case for x in data.index]
                best_labels = [x in best_case for x in data.index]
                worst = f1_score(labels, worst_labels)
                best = f1_score(labels, best_labels)
                operations.append((child.name(), worst, best))
    return operations

def get_add_np_disj_operations(theory, data,labels):

    operations=[]
    data["label"]=labels

    original_indices = theory.get_covering_examples(data)
    other_indices=set(data.index)-original_indices
    other_data=data.loc[list(other_indices)]
    positives=set(other_data.query("label==1").index)
    negatives=other_indices-positives

    #If the theory is a single var or a conjunction, add a disjunction on top
    if theory.is_var() or theory.is_and():
        worst_case=theory.get_covering_examples(data).union(negatives)
        best_case = theory.get_covering_examples(data).union(positives)
        worst_labels = [x in worst_case for x in data.index]
        best_labels = [x in best_case for x in data.index]
        worst = f1_score(labels, worst_labels)
        best = f1_score(labels, best_labels)
        operations.append(("AD"+theory.name(), worst, best))

    #Only select disjunctions that don't have child or parent disjunction that already contain a neural predicate
    disjunctions = [node for node in theory.nodes() if node.is_or()]
    allowed_disjunctions=[]
    for disj in disjunctions:
        if adding_np_is_allowed(theory, disj):
            allowed_disjunctions.append(disj)

    for i in allowed_disjunctions:
        #alter the init_flows such that worst and best case are put through the circuit
        worst_case = theory.get_covering_examples(data, [i.name(), negatives])
        best_case = theory.get_covering_examples(data, [i.name(), positives])

        worst_labels=[x in worst_case for x in data.index]
        best_labels = [x in best_case for x in data.index]
        worst=f1_score(labels, worst_labels)
        best=f1_score(labels, best_labels)
        operations.append((i.name(), worst, best))


    #For all VARS in conjunction, add a neural predicate in disjunction
    conjunctions = [node for node in theory.nodes() if node.is_and()]

    for i in conjunctions:
        for child in i.children:
            if child.is_var():# and adding_np_is_allowed(theory, child):
                worst_case = theory.get_covering_examples(data, [child.name(), negatives, False])
                best_case = theory.get_covering_examples(data, [child.name(), positives, False])
                worst_labels = [x in worst_case for x in data.index]
                best_labels = [x in best_case for x in data.index]
                worst = f1_score(labels, worst_labels)
                best = f1_score(labels, best_labels)
                operations.append((child.name(), worst, best))
    return operations

def get_ordered_operations(theory, data, labels, alpha=.75, estimate="sharpe"):
    print("Setting covering examples")
    theory.set_covering_examples(data)
    all_operations=[]
    print("Getting removal operations")
    rem_operations=sorted(get_removal_operations(theory, data, labels),key=lambda x : x[1], reverse=True)
    print("Start getting conjunctions")
    O_r=rem_operations[0] #Used in the Sharpe ratio to score the neural predicates
    conj_np=get_add_np_conj_operations(theory, data, labels)
    print("-->", len(conj_np), "conjunctions")
    disj_np=get_add_np_disj_operations(theory, data, labels)
    print("-->", len(disj_np), "disjunctions")
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
                sharpe=(alpha*B+(1-alpha)*W-O_r[1])/math.sqrt(alpha*((1-alpha)*B-(1-alpha)*W)**2+(1-alpha)*(alpha*W-alpha*B)**2)
            all_operations.append((name, sharpe, W, B))
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

    return all_operations, rem_operations

def get_removal_operations(theory, data, labels):
    # print("GET REMOVAL: ", len(theory.get_covering_examples()))
    all_operations=[("O",f1_score(labels, [x in theory.get_covering_examples() for x in data.index]))]
    all_vars = theory.variables() + theory.neural_predicates()
    if len(all_vars)>1:
        for var in all_vars:
            new_theory=remove_variables(theory, [var.name()])
            # print("REMOVED ", var, len(new_theory.get_covering_examples()))
            # new_theory.get_covering_examples(data)
            # print(len(new_theory.variables()))
            all_operations.append(("R"+var.name(), f1_score(labels, [x in new_theory.get_covering_examples() for x in data.index])))
    return all_operations




