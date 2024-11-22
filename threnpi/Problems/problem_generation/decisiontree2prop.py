from threnpi.src.propositional import VAR, OR, AND

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
            conditions[condition] = VAR(len(conditions), condition, data)

        if len(lines)==0:
            return conditions[condition], counters, conditions
        else:
            sub_result, counters, conditions=decisiontree2prop(lines, counters, conditions, data)
            new_conj=AND(counters[0],[conditions[condition], sub_result])
            counters[0]+=1
            return new_conj, counters, conditions

# def decisiontree2prop2(lines, c_counter=0, vars={}, start=True):
#     line=lines.pop(0) #head
#     depth=line.count("|")
#     same_depth=[line]+[x for x in lines if x.count("|")==depth]
#     if len(same_depth)>1: #disjunction
#         result={}
#         disj=[]
#         while len(same_depth)>0:
#            head=same_depth.pop(0)
#            if ":" in head:
#              condition=head[4*depth:head.find(":")]
#            else:
#                var = head[4 * depth:]
#            if var not in vars:
#                vars[var]="V"+str(len(vars))
#            if len(same_depth)>0:
#                subs=lines[:lines.index(same_depth[0])]
#                lines=lines[lines.index(same_depth[0])+1:]
#            else:
#                subs=lines
#                lines=[]
#            if len(subs)==0 and head.strip()[-1]!=":": #disjunction of 2 variables
#                disj.append([vars[var]])
#            elif len(subs)>0:
#                sub_result, c_counter, vars=decisiontree2prop(subs, c_counter, vars, False)
#                sub_head=sub_result.pop("P")
#                result.update(sub_result)
#                disj.append([vars[var]]+ sub_head)
#            else:
#                 vars.pop(var)
#         result.update({"P":["C"+str(c_counter)], "C"+str(c_counter):disj})
#         if start:
#              for x in result:
#                 if type(result[x])==list:
#                     if len([y for y in result[x] if type(y)==list])>0:
#                         new_list=[]
#                         for y in result[x]:
#                             if type(y)==list:
#                                 new_list.append(y)
#                             else:
#                                 new_list.append([y])
#                         result[x]=new_list
#         result.update({vars[x]:x.replace(",",".") for x in vars})
#         return result, c_counter+1, vars
#     else:
#         if ":" in line:
#             var=line[4*depth:line.find(":")]
#         else:
#             var = line[4 * depth:]
#         if var not in vars:
#             new_var="V"+str(len(vars))
#             vars[var]=new_var
#         else:
#             new_var=vars[var]
#         if len(lines)==0:
#             return {"P":[new_var],new_var:var }, c_counter, vars
#         else:
#             result, c_counter, vars=decisiontree2prop(lines, c_counter, vars, False)
#             result["P"]=[vars[var]]+result["P"]
#             if start:
#              for x in result:
#                     if type(result[x])==list:
#                         if len([y for y in result[x] if type(y)==list])>0:
#                             new_list=[]
#                             for y in result[x]:
#                                 if type(y)==list:
#                                     new_list.append(y)
#                                 else:
#                                     new_list.append([y])
#                             result[x]=new_list
#             result.update({vars[x]:x.replace(",",".") for x in vars})
#             return result, c_counter, vars
#
