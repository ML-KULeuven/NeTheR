import torch
import torch.nn as nn
from torchvision.models import resnet50,  ResNet50_Weights
import torch.optim as optim
import pytorch_lightning as L
import math

class mlp(L.LightningModule):
    def __init__(self, pos_weight, input_dim=518):
        super().__init__()
        self.input_dim=input_dim
        self.model=nn.Sequential(
            nn.Linear(input_dim, 2*input_dim),
            nn.ReLU(),
            nn.Linear(2*input_dim, 2*input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*input_dim, 2*input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))


    def forward(self, inputs):

        if type(inputs) == list and len(inputs) == 1:
            inputs = inputs[0]
        subr=self.model(inputs[:,-self.input_dim:]).flatten()

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
