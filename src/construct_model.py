from pysdd.sdd import SddManager,Vtree,WmcManager
from pysdd.iterator import SddIterator
import torch
import torch.nn as nn
import pytorch_lightning as L
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet18, ResNet18_Weights
import torchtime.models as time_models


def sdd_eval_lightning(manager, root_node,node2id, tensors, bools):
    iterator = SddIterator(manager, smooth=False)

    tensor_values = [value for value in tensors if isinstance(value, torch.Tensor)]
    bool_values = [value for value in bools.values() if isinstance(value, torch.Tensor)]
    reference = next(iter(tensor_values + bool_values), None)
    target_device = reference.device if reference is not None else torch.device("cpu")

    def as_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.to(target_device)
        return torch.as_tensor(value, dtype=torch.float32, device=target_device)

    def _formula_evaluator(node, r_values, *_):
        if node is not None:
            if node.is_literal():
                literal = node.literal
                if abs(literal) not in bools:
                    if literal<0:
                        return torch.add(torch.neg(tensors[node2id[abs(literal)]]), 1)
                    else:
                        return tensors[node2id[node.literal]]
                else:
                    if literal<0:
                        return 1-bools[abs(literal)]
                    else:
                        return bools[literal]
            elif node.is_true():
                return 1.0
            elif node.is_false():
                return 0.0
        # print("R_values: ", r_values, "-->", torch.stack([torch.mul(value[0], value[1]) for value in r_values]).sum(dim=0))
        return torch.stack(
            [as_tensor(value[0]) * as_tensor(value[1]) for value in r_values]
        ).sum()

    result = iterator.depth_first(root_node, _formula_evaluator)

    return result

class KClayer_lightning(L.LightningModule):
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
                    if mode == "animal_conv" or mode=="medical_diagnosis":

                        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)  # You can use resnet50, resnet101, etc.
                        # Remove the fully connected layer
                        if mode=="medical_diagnosis":
                            # Modify the first convolutional layer to accept 6 input channels
                            old_conv1 = resnet.conv1
                            resnet.conv1 = nn.Conv2d(
                                in_channels=6,  # Change to match your input
                                out_channels=old_conv1.out_channels,
                                kernel_size=old_conv1.kernel_size,
                                stride=old_conv1.stride,
                                padding=old_conv1.padding,
                                bias=old_conv1.bias
                            )

                            # Copy the pretrained weights for the first 3 channels
                            resnet.conv1.weight.data[:, :3, :, :] = old_conv1.weight.data
                            resnet.conv1.weight.data[:, 3:, :, :] = old_conv1.weight.data

                        num_ftrs = resnet.fc.in_features
                        resnet.fc = nn.Identity()  # Remove the final classification layer
                        for param in resnet.parameters():
                            param.requires_grad = False
                        # Define custom classification layers
                        neural_layers[var.id]=nn.Sequential(
                            # nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
                            resnet,
                            nn.Linear(num_ftrs, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            # nn.Linear(2048, 4096),
                            # nn.ReLU(),
                            # nn.Linear(4096, 512),
                            # nn.ReLU(),
                            # nn.Linear(512, 256),
                            # nn.ReLU(),
                            # nn.Linear(256, 128),
                            # nn.ReLU(),
                            nn.Linear(256, 1),
                            # nn.ReLU(),
                            # nn.Linear(518, 1)
                        )
                    elif mode=="light_conv" or mode=="motif_conv":
                       neural_layers[var.id]=time_models.InceptionTime(n_inputs=1, n_classes=1)
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
        elif self.mode == "medical_diagnosis":
            reshaped = inputs[:, -6 * 512 * 512:].reshape(len(inputs), 6, 512, 512)
            neural_results = [torch.sigmoid(x.forward(reshaped)) if x != None else None for x in self.neural_layers]

        elif self.mode == "light_conv":
            reshaped = inputs[:, -1024:]
            neural_results=[]
            for x in self.neural_layers:
                if x!=None:
                    neural_results.append(x.forward(reshaped.unsqueeze(1)).flatten())
                else:
                    neural_results.append(None)
        elif self.mode == "motif_conv":
            reshaped = inputs[:, -510:]
            neural_results = []
            for x in self.neural_layers:
                if x != None:
                    neural_results.append(x.forward(reshaped.unsqueeze(1)).flatten())
                else:
                    neural_results.append(None)
            # print("\nNeural results:",neural_results)
        else:
            neural_results =  [torch.sigmoid(x.forward(inputs[:,-518:])) if x!=None else None for x in self.neural_layers]
        logic_results = {x:self.logic_layers[x](inputs) for x in self.logic_layers}

        result = []
        for i in range(0, len(inputs)):
            # print("NEURAL INPUTS: ", [x[i].squeeze() if x!=None else None for x in neural_results])
            # print("LOGIC INPUTS: ", {x.literal:logic_results[x][i] for x in logic_results})
            res = sdd_eval_lightning(self.manager, self.circuit, self.node2id,[x[i].squeeze() if x!=None else None for x in neural_results],
                                     {x.literal:logic_results[x][i] for x in logic_results})

            new_res = torch.logit(res.clamp(0.01, 0.99))
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
        return torch.optim.Adam(self.parameters(), lr=0.0001)



def generate_logiccondition(condition, columns):
    if "<=" in condition:
        return lambda z: [float(
            x[columns.index(condition[:condition.find("<=")].strip())] <= float(condition[condition.find("<=") + 2:]))
                          for x in z]
    elif ">=" in condition:
        return lambda z: [float(
            x[columns.index(condition[:condition.find(">=")].strip())] >= float(condition[condition.find(">=") + 2:]))
                          for x in z]
    elif "<" in condition:
        return lambda z: [float(
            x[columns.index(condition[:condition.find("<")].strip())] < float(condition[condition.find("<") + 1:])) for
                          x in z]
    elif ">" in condition:
        return lambda z: [float(
            x[columns.index(condition[:condition.find(">")].strip())] > float(condition[condition.find(">") + 1:])) for
                          x in z]
    else:
        return lambda z: [float(
            x[columns.index(condition[:condition.find("==")].strip())] == float(condition[condition.find("==") + 2:]))
            for x in z]

def prop2sdd(theory, var_dict):
    if theory.is_var() or theory.is_neural_predicate():
        return var_dict[theory]
    elif theory.is_or():
        result=prop2sdd(theory.children[0], var_dict)
        for child in theory.children[1:]:
            result=(result | prop2sdd(child, var_dict))
        return result
    else:
        result = prop2sdd(theory.children[0], var_dict)
        for child in theory.children[1:]:
            result = (result & prop2sdd(child, var_dict))
        return result


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

    model = KClayer_lightning(sdd_form, manager, var_dict, columns, 0.1, pos_weight, mode, freeze)
    return model

