import torch
import torch.nn as nn
from torchvision.models import resnet50,  ResNet50_Weights
import torch.optim as optim
import pytorch_lightning as L
import math

class res_cnn(L.LightningModule):
    def __init__(self, weight, mode="animal_conv"):
        super().__init__()
        self.mode=mode
        # Load pretrained ResNet model

        self.pre_layer = nn.Sequential(
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        )

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # You can use resnet50, resnet101, etc.
        # Remove the fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the final classification layer
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Define custom classification layers
        self.classifier = nn.Sequential(
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
            nn.Linear(64, 1))

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))

    def forward(self, x):
        if type(x) == list and len(x) == 1:
            x = x[0]
        # Extract features using ResNet
        if self.mode=="animal_conv":
            x = self.pre_layer(x[:, -64 * 64 * 3:].reshape(len(x), 3, 64, 64))  # Adjust input if necessary
        else:
            x = self.pre_layer(x[:, -512 * 512 * 3:].reshape(len(x), 3, 512, 512))  # Adjust input if necessary

        features = self.resnet(x)
        # Classify features
        logits = self.classifier(features)
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


class multimodal_cnn(L.LightningModule):
    def __init__(self, num_tab, weight, mode="animal_conv"):
        super().__init__()
        print("Num tabular: ", num_tab)
        # Load pretrained ResNet model
        self.pre_layer = nn.Sequential(
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        )
        self.mode=mode
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # You can use resnet50, resnet101, etc.
        # Remove the fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the final classification layer
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Define custom classification layers
        self.classifier = nn.Sequential(
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
            nn.Linear(64, 5))
        self.mlp = nn.Sequential(
            nn.Linear(num_tab, int(math.ceil(num_tab*2))),
            nn.ReLU(),
            nn.Linear(int(math.ceil(num_tab*2)), int(math.ceil(num_tab*2))),
            nn.ReLU(),
            nn.Linear(int(math.ceil(num_tab*2)), int(math.ceil(num_tab))),
            nn.ReLU(),
            nn.Linear(int(math.ceil(num_tab)), 5))
        self.relu=nn.ReLU()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
        self.finish=nn.Linear(10,1)
    def forward(self, x):
        if type(x) == list and len(x) == 1:
            x = x[0]
        # Extract features using ResNet
        if self.mode=="animal_conv":
            sub_res = self.pre_layer(x[:, -64 * 64 * 3:].reshape(len(x), 3, 64, 64))  # Adjust input if necessary
        else:
            sub_res = self.pre_layer(x[:, -512 * 512 * 3:].reshape(len(x), 3, 512, 512))  # Adjust input if necessary

        features = self.resnet(sub_res)
        # Classify features
        image_logits = self.classifier(features)
        if self.mode=="animal_conv":
            mlp_logits=self.mlp(x[:, :-64 * 64 * 3])
        else:
            mlp_logits = self.mlp(x[:, :-512 * 512 * 3])
        logits = torch.cat((image_logits, mlp_logits), dim=1)
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