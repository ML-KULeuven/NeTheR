
from pysdd.sdd import Vtree, SddManager, SddNode
import pandas as pd


from nether.src.construct_model import prop2sdd
from nether.src.neural_architectures.convolutional import res_cnn, multimodal_cnn
from nether.src.propositional import VAR, OR, AND
from sklearn import tree
from sklearn.tree import export_text
from nether.src.propositional import clean_theory

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


def decisiontree2prop(lines, counters, conditions,data):
    line = lines.pop(0)  # head
    depth = line.count("|")
    same_depth = [line] + [x for x in lines if x.count("|") == depth]
    if len(same_depth) > 1:  # disjunction
        disj = []
        while len(same_depth) > 0:
            head = same_depth.pop(0)
            if ":" in head:
                condition = head[4 * depth:head.find(":")]
            else:
                condition = head[4 * depth:]
            if condition not in conditions:
                # print("CONDITION: ", condition)
                conditions[condition] = VAR(len(conditions), condition, data)
            if len(same_depth) > 0:
                subs = lines[:lines.index(same_depth[0])]
                lines = lines[lines.index(same_depth[0]) + 1:]
            else:
                subs = lines
                lines = []
            if len(subs) == 0 and head.strip()[-1] != ":":  # disjunction of 2 variables
                disj.append(conditions[condition])
            elif len(subs) > 0:
                sub_result, counters,conditions = decisiontree2prop(subs, counters, conditions, data)
                new_conj=AND(counters[0],[conditions[condition], sub_result])
                counters[0]+=1
                disj.append(new_conj)
            else:
                conditions.pop(condition)
        new_disj=OR(counters[1], disj)
        counters[1]+=1
        return new_disj, counters, conditions
    else:
        if ":" in line:
            condition=line[4*depth:line.find(":")]
        else:
            condition = line[4 * depth:]
        if condition not in conditions:
            # print("CONDITION: ", condition)
            conditions[condition] = VAR(len(conditions), condition, data)

        if len(lines)==0:
            return conditions[condition], counters, conditions
        else:
            sub_result, counters, conditions=decisiontree2prop(lines, counters, conditions, data)
            new_conj=AND(counters[0],[conditions[condition], sub_result])
            counters[0]+=1
            return new_conj, counters, conditions

def learn_decision_tree(train_data,train_labels, cat_cols, min_split=0, min_leaf=0, depth=0, single_split=False):
    train_data=train_data[cat_cols]
    if len(train_data)==0:
        return None, None
    # train_data["label"]=[str(x) for x in train_labels]
    if single_split:
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=1, class_weight="balanced")
    else:
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth, min_samples_split=min_split,
                                          min_samples_leaf=min_leaf, class_weight="balanced")

    # max_depth = 10, min_samples_split = 5, min_samples_leaf = 2
    clf = clf.fit(train_data, train_labels)

    dt_string=export_text(clf, max_depth=1000, feature_names=cat_cols)
    lines=dt_string.split("\n")
    zero_class=[x for x in range(0,len(lines)) if "class: 0" in lines[x]]

    new_lines=[]
    for x in range(0,len(lines)):
        if x not in zero_class and x+1 not in zero_class:
            new_lines.append(lines[x].replace("|---",""))

    one_class = [x for x in range(0, len(new_lines)) if "class: 1" in new_lines[x]]
    final_lines = []
    for x in range(0, len(new_lines)):
        if x + 1 in one_class:
            final_lines.append(new_lines[x] + ": 1")
        elif x not in one_class:
            final_lines.append(new_lines[x])
    # print("##################")
    # for x in final_lines:
    #     print(x)
    to_rem=[]
    for x in range(0,len(final_lines)-1):
        if final_lines[x].count("|")>=final_lines[x+1].count("|") and ": 1" not in final_lines[x]:
            to_rem.append(x)
    while len(to_rem)>0:
        # print("TO_REM: ", to_rem)
        for x in sorted(to_rem, reverse=True):
            final_lines.pop(x)
        to_rem=[]
        for x in range(0, len(final_lines) - 1):
            if final_lines[x].count("|") >= final_lines[x + 1].count("|") and ": 1" not in final_lines[x]:
                to_rem.append(x)
    final_lines=[x for x in final_lines if len(x)>0]
    # print("-------------")
    # for i in final_lines:
    #     print(i)
    # print("--------------")
    if len(final_lines)>0:
        dt,_,_=decisiontree2prop(final_lines, [0,0], {}, train_data)
        dt=clean_theory(dt)
    else:
        return None, None
    return dt, clf