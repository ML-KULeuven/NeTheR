from nether.src.train_model import validate_model
import torch.nn as nn
import pytorch_lightning as L
import torch
import copy

class PROPOSITIONAL:
    def __init__(self, id):
        self.id = id
        self.covering_examples = []
        self.circuit_flow = set()

    def id(self):
        return self.id

    def get_circuit_flow(self):
        return self.circuit_flow

    def set_circuit_flow(self, data):
        pass

    def name(self):
        pass

    def get_covering_examples(self,data=None, alteration=None, device=None):
        pass

    def set_covering_examples(self, data, evice=None):
        pass

    def validate(self):
        pass

    def show(self):
        pass

    #Return all VAR nodes
    def variables(self):
        pass

    def neural_predicates(self):
        pass
    #Return all nodes
    def nodes(self):
        pass

    def save(self):
        pass

    def is_or(self):
        return False

    def is_and(self):
        return False

    def is_var(self):
        return False

    def is_neural_predicate(self):
        return False

    def dnf(self):
        pass

class VAR(PROPOSITIONAL):
    def __init__(self, id, condition, data=None, indices=None):
        super().__init__(id)
        self.condition = condition.replace(",",".")
        if indices!=None:
            self.covering_examples=indices
        elif "None" in str(type(data)) and indices==None:
            self.covering_examples=set()
        else:
            self.covering_examples=set(data.query(self.condition).index)

    def set_covering_examples(self, data, device=None):
        self.covering_examples = set(data.query(self.condition).index)

    def get_covering_examples(self,data=None, alteration=None, device=None):

        if "None" not in str(type(data)):
            all_operators=["==",">=","<=",">","<"]
            operator=all_operators[[x in self.condition for x in all_operators].index(True)]
            self.covering_examples = set(
                data[[self.condition[:self.condition.find(operator)].strip()]].query(self.condition).index)

        if alteration !=None and alteration[0]==self.name():
            #In disjunction --> get results for conjunction
            if alteration[2]:
                return self.covering_examples.intersection(alteration[1])
            else:
                return self.covering_examples.union(alteration[1])
        return self.covering_examples

    def name(self):
        return "V"+str(self.id)

    def validate(self):
        return True

    def set_circuit_flow(self, indices):
        self.circuit_flow=indices.intersection(self.get_covering_examples())

    def show(self):
        return self.name()+" : "+self.condition+"\n"

    def variables(self):
        return [self]

    def neural_predicates(self):
        return []

    def nodes(self):
        return [self]

    def save(self):
        return {self.name():self.condition}

    def is_var(self):
        return True

    def dnf(self):
        return self

class AND(PROPOSITIONAL):
    def __init__(self, id, children,device=[0,]):
        super().__init__(id)
        self.children = children
        if len(children)>0:
            result=self.children[0].get_covering_examples()
            for child in self.children[1:]:
                result = result.intersection(child.get_covering_examples(device=device))
            self.covering_examples=result
        else:
            self.covering_examples=set()

    def set_covering_examples(self, data, device):
        # print("AND set covering: ", device)
        result = set(data.index)
        for child in self.children:
            result = result.intersection(child.get_covering_examples(data, device=device))
        self.covering_examples = result

    def get_covering_examples(self,data=None, alteration=None, device=None):

            if "None" not in str(type(data)):
                result=set(data.index)
                for child in self.children:
                    result=result.intersection(child.get_covering_examples(data, alteration, device=device))
                self.covering_examples=result
            if alteration!=None and alteration[0]==self.name():
                return self.covering_examples.intersection(alteration[1])
            return self.covering_examples

    def add_child(self, child):
        self.children.append(child)

    def validate(self):
        if len(self.children)<2:
            return False
        for child in self.children:
            if not child.validate():
                return False
        if len([x for x in self.children if x.is_and()])!=0:
            return False
        return True

    def name(self):
        return "C"+str(self.id)

    def show(self):
        result=""
        conj=""
        for child in self.children:
            result+=child.show()
            conj+=child.name() +" && "
        return result+self.name()+" : "+conj[:-4]+"\n"

    def set_circuit_flow(self, indices):
        ## propogate the indices to the children
        self.circuit_flow=indices.intersection(self.get_covering_examples())
        for child in self.children:
            child.set_circuit_flow(self.circuit_flow)


    def variables(self):
        vars=[]
        for child in self.children:
            vars+=child.variables()
        return vars

    def neural_predicates(self):
        preds = []
        for child in self.children:
            preds += child.neural_predicates()
        return preds

    def nodes(self):
        result=[self]
        for child in self.children:
            result+=child.nodes()
        return result



    def save(self):
        result={}
        children=[]
        for child in self.children:
            result.update(child.save())
            children.append(child.name())
        result[self.name()]=children
        return result

    def is_and(self):
        return True

class OR(PROPOSITIONAL):
    def __init__(self, id, children, device=[0,]):
        super().__init__(id)
        self.children = children
        result = set()
        if len(children)>0:
            for child in self.children:
                result = result.union(child.get_covering_examples(device=device))
            self.covering_examples=result
        else:
            self.covering_examples=set()

    def set_covering_examples(self, data, device):
        result = set()
        for child in self.children:
            result = result.union(child.get_covering_examples(data, device=device))
        self.covering_examples = result

    def get_covering_examples(self,data=None, alteration=None, device=None):
        if "None" not in str(type(data)):
            result=set()
            for child in self.children:
                result=result.union(child.get_covering_examples(data, alteration, device=device))
            self.covering_examples=result
        if alteration != None and alteration[0] == self.name():
            return self.covering_examples.union(alteration[1])
        return self.covering_examples

    def validate(self):
        if len(self.children) < 2:
            return False
        for child in self.children:
            if not child.validate():
                return False
        if len([x for x in self.children if x.is_or()])!=0:
            return False
        return True

    def add_child(self, child):
        self.children.append(child)

    def name(self):
        return "D"+str(self.id)

    def show(self):
        result=""
        disj=""
        for child in self.children:
            result+=child.show()
            disj+=child.name() +" | "
        return result+self.name()+" : "+disj[:-3]+"\n"

    def set_circuit_flow(self, indices):
        ## propogate the indices to the children
        self.circuit_flow=indices.intersection(self.get_covering_examples())
        for child in self.children:
            child.set_circuit_flow(self.circuit_flow)

    def variables(self):
        vars=[]
        for child in self.children:
            vars+=child.variables()
        return vars

    def neural_predicates(self):
        preds = []
        for child in self.children:
            preds += child.neural_predicates()
        return preds

    def nodes(self):
        result = [self]
        for child in self.children:
            result += child.nodes()
        return result

    def save(self):
        result={}
        children=[]
        for child in self.children:
            result.update(child.save())
            children.append(child.name())
        result[self.name()]=children
        return result

    def is_or(self):
        return True

    def dnf(self):
        new_children=[x.dnf() for x in self.children]
        return OR(self.id, new_children)

class NEURAL_PREDICATE(PROPOSITIONAL):
    def __init__(self, id, mode, neural_layer=None, indices=set()):
        super().__init__(id)
        self.neural_layer=neural_layer
        self.covering_examples=indices
        self.mode=mode
        self.last_indices=set()


    def validate(self):
        return True

    def name(self):
        return "NN"+str(self.id)

    def show(self):
        return self.name()+"\n"

    def variables(self):
        return []

    def neural_predicates(self):
        return [self]

    def nodes(self):
        return [self]

    def save(self):
        return {self.name():"NN"}

    def get_covering_examples(self,data=None, alteration=None, device=None):
        if "None" not in str(type(data)) and set(data.index)!=self.last_indices:
            self.last_indices=set(data.index)
            sub=set()

            labels=validate_model(val_model(self.neural_layer, self.mode), data, device=device)
            for x in data.index:
                if labels.pop(0)==1:
                    sub.add(x)
            self.covering_examples=sub
        return self.covering_examples

    def set_covering_examples(self, data, device):
        sub = set()
        labels = validate_model(val_model(self.neural_layer, self.mode), data, device=device)
        for x in data.index:
            if labels.pop(0) == 1:
                sub.add(x)
        self.covering_examples = sub

    def is_neural_predicate(self):
        return True




def clean_theory(theory:PROPOSITIONAL):
    if theory==None:
        return None
    if theory.validate(): #automatically happens with var and NP
        return theory
    else:
        new_children=[x for x in [clean_theory(child) for child in theory.children] if x!=[]]
        if len(new_children)==0:
            return []
        elif len(new_children)==1:
            return new_children[0]
        elif theory.is_and():
            if len([x for x in new_children if x.is_and()])!=0:
                and_children=[]
                for child in new_children:
                    if child.is_and():
                        and_children+=child.children
                    else:
                        and_children.append(child)
                return AND(theory.id, and_children)
            else:
                return AND(theory.id, new_children)
        else:
            if len([x for x in new_children if x.is_or()]) != 0:
                or_children = []
                for child in new_children:
                    if child.is_or():
                        or_children += child.children
                    else:
                        or_children.append(child)
                return OR(theory.id, or_children)
            else:
                return OR(theory.id, new_children)

def load_theory(theory, head=None):
    #load all the vars
    if head == None:
        ops = [x for x in theory]
        gen_children = []
        for op in ops:
            if op[0]!="V" and op[0]!="N":
                gen_children += theory[op]
        head = (set(ops) - set(gen_children)).pop()
    if head[0]=="V":
        id = int(head[1:])
        condition = theory[head]
        return VAR(id, condition, indices=set())
    elif head[0] == "N":
        id = int(head[2:])
        return NEURAL_PREDICATE(id, "animal_conv")
    else:
        children=[]
        for child in theory[head]:
            children.append(load_theory(theory, child))
        if head[0]=="C":
            return AND(int(head[1:]), children)
        else:
            return OR(int(head[1:]), children)

class val_model(L.LightningModule):
    def __init__(self,neural_layer, mode):
        super().__init__()

        self.mode=mode
        self.neural_layers = nn.ModuleList([neural_layer])
        self.bce = torch.nn.BCEWithLogitsLoss()


    def forward(self, inputs):
        if type(inputs) == list and len(inputs) == 1:
            inputs = inputs[0]
        if self.mode == "house_conv":
            reshaped = inputs[:, -2048 * 3:].reshape(len(inputs), 3, 32, 64)
            neural_results = [x.forward(reshaped) if x!=None else None for x in self.neural_layers]
        elif self.mode == "mnist_conv":
            reshaped = inputs[:, -28 * 28:].reshape(len(inputs), 1, 28, 28)
            neural_results =  [x.forward(reshaped) if x!=None else None for x in self.neural_layers]
        elif self.mode == "animal_conv":
            reshaped = inputs[:, -3 * 64 * 64:].reshape(len(inputs), 3, 64, 64)
            neural_results =  [torch.sigmoid(x.forward(reshaped)) if x!=None else None for x in self.neural_layers]
        elif self.mode == "medical_diagnosis":
            reshaped = inputs[:, -6 * 512 * 512:].reshape(len(inputs), 6, 512, 512)
            neural_results = [torch.sigmoid(x.forward(reshaped)) if x != None else None for x in self.neural_layers]

        elif self.mode=="light_conv":
            reshaped = inputs[:, -1024:]
            neural_results = []
            for x in self.neural_layers:
                neural_results.append(x.forward(reshaped.unsqueeze(1)).flatten())
        elif self.mode=="motif_conv":
            reshaped = inputs[:, -510:]
            neural_results = []
            for x in self.neural_layers:
                neural_results.append(x.forward(reshaped.unsqueeze(1)).flatten())

        else:
            neural_results =  [x.forward(inputs[:,-518:]) if x!=None else None for x in self.neural_layers]
        result = []
        for i in range(0, len(inputs)):
            # print("NEURAL INPUTS: ", [x[i].squeeze() if x!=None else None for x in neural_results])
            # print("LOGIC INPUTS: ", {x.literal:logic_results[x][i] for x in logic_results})
            res=[x[i].squeeze() if x!=None else None for x in neural_results][0]
            new_res = torch.logit(res.clamp(0.01, 0.99))
            # print("RESULT -->", new_res)
            result.append(new_res)
        result = torch.stack(result).flatten()
        # print("VAL MODEL RESULT = ", result)
        self.last_res = result
        if result.requires_grad:
            self.last_res.retain_grad()
        return result

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.bce(output, target)
        self.log("training_loss", loss, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.bce(output, target)
        self.log("val_loss", loss, prog_bar=True, logger=False)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.bce(output, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00001)


def remove_variables(theory, to_remove,device=[0,]):
    theory=copy.deepcopy(theory)
    if theory.is_var():
        if theory.name() in to_remove:
            return None
        else:
            return VAR(theory.id, theory.condition, indices=theory.get_covering_examples(device=device))
    elif theory.is_neural_predicate():
        if theory.name() in to_remove:
            return None
        else:
            return NEURAL_PREDICATE(theory.id, theory.mode, theory.neural_layer, indices=theory.get_covering_examples(device=device))
    else:
        new_children=[x for x in [remove_variables(child, to_remove,device=device) for child in theory.children] if x!=None]
        if theory.is_and():
            return clean_theory(AND(theory.id, new_children))
        else:
            return clean_theory(OR(theory.id, new_children))
