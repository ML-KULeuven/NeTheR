from nether.src.propositional import AND, OR, VAR

def rules2prop(rules, data):
    if (rules.count("Labels=1")==1 and rules.count("Labels=0")==0) or (rules.count("Labels=1")==0 and rules.count("Labels=0")==1):
        return None
    conditions={}
    if rules.count("Labels=1")>rules.count("Labels=0"):
        disj=[]
        for rule in [x[:x.find("=> Labels")] for x in rules.split("\n")[3:] if "Labels=1" in x]:
            conj=[]
            for condition in rule.split(" and "):
                condition=condition.strip()[1:-1]
                if condition not in conditions:
                    conditions[condition]=VAR(len(conditions), condition, data)
                conj.append(conditions[condition])
            new_conj=AND(len(disj), conj)
            disj.append(new_conj)
        new_disj=OR(0, disj)
        return new_disj
    elif rules.count("Labels=1")==rules.count("Labels=0"):
        splitted=[x for x in rules.split("\n")[3:-3]]
        if len(splitted[0])<len(splitted[1]):
            longest=splitted[1]
        else:
            longest=splitted[0]
        if "=> Labels=0" in longest:
            disj=[]
            for condition in longest[:longest.find("=> Labels")].split(" and "):
                condition=condition.strip()[1:-1]
                if "==" in condition:
                    print("HELP")
                if ">" in condition:
                    condition=condition.replace(">=","<")
                else:
                    condition=condition.replace("<=",">")
                if condition not in conditions:
                    conditions[condition]=VAR(len(conditions), condition, data)
                disj.append(conditions[condition])
            return OR(0,disj)
        else:
            conj=[]
            for condition in longest[:longest.find("=> Labels")].split(" and "):
                condition=condition.strip()[1:-1]
                if condition not in conditions:
                    conditions[condition]=VAR(len(conditions), condition, data)
                conj.append(conditions[condition])
            return AND(0,conj)
    else:
        conj=[]

        for rule in [x[:x.find("=> Labels")] for x in rules.split("\n")[3:] if "Labels=0" in x]:
            disj=[]
            for condition in rule.split(" and "):
                condition=condition.strip()[1:-1]
                if "==" in condition:
                    print("HELP")
                if ">" in condition:
                    condition=condition.replace(">=","<")
                else:
                   condition=condition.replace("<=",">")
                if condition not in conditions:
                    conditions[condition]=VAR(len(conditions), condition, data)
                disj.append(conditions[condition])

            conj.append(OR(len(conj), disj))

        return AND(0,conj)

