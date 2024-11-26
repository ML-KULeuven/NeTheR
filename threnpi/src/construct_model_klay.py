import klay
from pysdd.sdd import SddManager,Vtree,WmcManager
from pysdd.iterator import SddIterator
import torch
import torch.nn as nn
import pytorch_lightning as L
from torchvision.models import resnet50, ResNet50_Weights
from construct_model import prop2sdd, generate_logiccondition

def klayers_eval(klayers_mod, node2id, tensors, bools):
    # setup weights
    num_vars = int(klayers_mod.layers[0].csr.shape[0] / 2 - 1)
    weights = [None] * num_vars
    for lit_id, weight in bools.items():
        weights[lit_id-1] = torch.tensor(weight).to("cuda")
    for lit_id, idx in node2id.items():
        weights[lit_id-1] = tensors[idx]

    # stack the tensors in the weights list into one large tensor
    weights = torch.stack(weights).to("cuda")
    weights = torch.log(weights)
    return torch.exp(klayers_mod(weights, eps=1e-7))


class Klayer_lightning(L.LightningModule):

    def __init__(self, circuit, manager, var_dict, columns, dropout, pos_weight, mode, freeze):
        super().__init__()
        self.circuit = circuit
        self.var_dict = var_dict
        self.mode = mode
        self.columns = columns
        self.dropout = dropout
        self.node2id={}
        max_id=0
        torch.set_grad_enabled(True)
        for i in [x for x in var_dict if x.is_neural_predicate()]:
            self.node2id[var_dict[i].literal]=i.id
            max_id=max(i.id+1, max_id+1)

        logic_layers = {}
        neural_layers = nn.ModuleList([None for x in range(0,max_id)])
        for var in var_dict:
            if var.is_var():
                func = generate_logiccondition(var.condition, list(columns))
                logic_layers[var_dict[var]]=func
            else:
                if var.neural_layer!=None:
                    neural_layers[var.id]=var.neural_layer
                    if freeze:
                        for param in neural_layers[var.id].parameters():
                            param.requires_grad = False
                else:
                    if mode == "animal_conv":

                        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # You can use resnet50, resnet101, etc.
                        # Remove the fully connected layer

                        num_ftrs = resnet.fc.in_features
                        resnet.fc = nn.Identity()  # Remove the final classification layer
                        for param in resnet.parameters():
                            param.requires_grad = False
                        # Define custom classification layers
                        neural_layers[var.id]=nn.Sequential(
                            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
                            resnet,
                            nn.Linear(num_ftrs, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 1)
                        )

                    else:
                        neural_layers[var.id]=nn.Sequential(
                            nn.Linear(518, 1036),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(1036, 1036),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(1036, 1024),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(512, 64),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(64, 1)
                        )
        self.logic_layers = logic_layers
        self.neural_layers = neural_layers
        self.manager = manager
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.last_res = None

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
        else:
            neural_results =  [x.forward(inputs[:,-518:]) if x!=None else None for x in self.neural_layers]
        logic_results = {x:self.logic_layers[x](inputs) for x in self.logic_layers}

        result = []
        for i in range(0, len(inputs)):
            # print("NEURAL INPUTS: ", [x[i].squeeze() if x!=None else None for x in neural_results])
            # print("LOGIC INPUTS: ", {x.literal:logic_results[x][i] for x in logic_results})
            res = klayers_eval(self.circuit, self.node2id,[x[i].squeeze() if x!=None else None for x in neural_results],
                                     {x.literal:logic_results[x][i] for x in logic_results})
            new_res = torch.logit(torch.maximum(torch.minimum(res, torch.tensor(0.99)), torch.tensor(0.01)))
            # print("RESULT -->", new_res)
            result.append(new_res)
        result = torch.stack(result).flatten()
        # print("RESULT = ", result)
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


def construct_model(theory, columns, pos_weight, mode, freeze):
    """

    :param theory: The propositional theory that has to be converted
    :param columns:
    :param pos_weight:
    :param mode:
    :return:
    """

    query_vars = theory.variables()+theory.neural_predicates()

    # set up vtree and manager
    var_count = len(query_vars)
    var_order = list(range(1, var_count + 1))
    vtree_type = "balanced"

    vtree = Vtree(var_count, var_order, vtree_type)
    manager = SddManager.from_vtree(vtree)

    var_list = [manager.literal(i) for i in range(1, var_count + 1)]
    var_dict = {query_vars[x]: var_list[x] for x in range(0, len(query_vars))}
    sdd_form = prop2sdd(theory, var_dict)
    sdd_form.ref()
    manager.minimize()

    klay_circuit = klay.Circuit()
    klay_circuit.add_sdd(sdd_form)
    klay_circuit_module = klay_circuit.to_torch_module().to("cuda")

    model = Klayer_lightning(klay_circuit_module, manager, var_dict, columns, 0.1, pos_weight, mode, freeze)
    return model

