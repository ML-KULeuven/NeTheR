
import torch
import torch.nn as nn
from torchvision.models import resnet50,  ResNet50_Weights
import torch.optim as optim
import pytorch_lightning as L
import math
import torchtime.models as models

class inception(L.LightningModule):
    def __init__(self, pos_weight, input_dim=1024):
        super().__init__()
        self.input_dim=input_dim
        self.model=models.InceptionTime(n_inputs=1, n_classes=1)


        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))


    def forward(self, inputs):

        if type(inputs) == list and len(inputs) == 1:
            inputs = inputs[0]
        subr=self.model(inputs[:,-self.input_dim:].unsqueeze(1)).flatten()

        return torch.logit(torch.maximum(torch.minimum(subr, torch.tensor(0.99)), torch.tensor(0.01)))


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


class multimodal_inception(L.LightningModule):
    def __init__(self, num_tab, weight, input_dim=1024):
        super().__init__()
        self.input_dim=input_dim
        self.mlp = nn.Sequential(
            nn.Linear(num_tab, int(math.ceil(num_tab*2))),
            nn.ReLU(),
            nn.Linear(int(math.ceil(num_tab*2)), int(math.ceil(num_tab*2))),
            nn.ReLU(),
            nn.Linear(int(math.ceil(num_tab*2)), int(math.ceil(num_tab))),
            nn.ReLU(),
            nn.Linear(int(math.ceil(num_tab)), 5))
        self.inception=models.InceptionTime(n_inputs=1, n_classes=5)
        self.relu=nn.ReLU()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
        self.finish=nn.Linear(10,1)

    def forward(self, x):
        if type(x) == list and len(x) == 1:
            x = x[0]
        # Extract features using ResNet


        # Classify features
        time_logits = self.inception(x[:,-self.input_dim:].unsqueeze(1))
        mlp_logits=self.mlp(x[:, :-self.input_dim])
        logits = torch.cat((time_logits, mlp_logits), dim=1)
        logits = self.finish(self.relu(logits))

        return logits

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.00001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.bce_loss(logits, y.unsqueeze(1).float())  # BCE loss for binary classification
        self.log('train_loss', loss, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.bce_loss(logits, y.unsqueeze(1).float())  # BCE loss for binary classification
        self.log('val_loss', loss, prog_bar=True, logger=False)
        return loss
