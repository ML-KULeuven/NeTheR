from threnpi.src.propositional import VAR, AND, OR, clean_theory
import random

def generate_random_theory(data, cat_columns, width, depth,intermediate_counters=[0,0,0]):
    num_disj = random.randint(1, width)
    all_cols=set()
    disj_children=[]
    for i in range(0,num_disj):
        num_conj = random.randint(1, width)
        conj_children=[]
        for x in range(0,num_conj):
            add_theory = bool(random.randint(1, 100) < 50) and depth > 1
            #add a conjunction or disjunction with 50% chance, otherwise add a variable
            if add_theory:
                new_theory, intermediate_counters, new_vars=generate_random_theory(data, cat_columns, width, depth-1, intermediate_counters)
                all_cols=all_cols.union(new_vars)
                conj_children.append(new_theory)
            else:
                done = False
                while not done:
                    rand_col = random.choice(cat_columns)
                    sign = random.choice([">", "<", "=="])
                    all_vals = set(data[rand_col])
                    min_val = int(min(all_vals))
                    max_val = int(max(all_vals))
                    if len(all_vals) < 3:
                        sign = " == "
                    if min_val != max_val:
                        if sign == " > ":
                            val = random.sample(list(all_vals - {max_val}), 1)[0]
                        elif sign == " < ":
                            val = random.sample(list(all_vals - {min_val}), 1)[0]
                        else:
                            val = random.sample(list(all_vals), 1)[0]
                        all_cols.add(rand_col)
                        new_var=VAR(intermediate_counters[0],rand_col + sign + str(val),data)
                        intermediate_counters[0]+=1
                        conj_children.append(new_var)
                        done=True
        new_child=AND(intermediate_counters[1], conj_children)
        intermediate_counters[1]+=1
        disj_children.append(new_child)
    result=OR(intermediate_counters[2], disj_children)
    intermediate_counters[2]+=1
    return clean_theory(result), intermediate_counters, all_cols


