from torch.utils.data import DataLoader, TensorDataset
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as L
from pytorch_lightning.callbacks import TQDMProgressBar

import time
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
import sys



def train_model(model, data,labels,  epochs, single=False):
    if not single:
        X_train, X_test, Y_train, Y_test = train_test_split(np.array(data, dtype=np.float64), np.array(labels, dtype=np.float64), test_size=0.1,random_state=42)
    else:
        X_train=np.array(data[0], dtype=np.float64)
        X_test=np.array(data[1], dtype=np.float64)
        Y_train=np.array(labels[0], dtype=np.float64)
        Y_test=np.array(labels[1], dtype=np.float64)
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
    val_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).float())
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # logger=TensorBoardLogger('tb_logs', default_hp_metric=False,log_graph=False)
    trainer = L.Trainer(max_epochs=epochs, devices=[0,], callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)])
    trainer.fit(model, train_loader, val_loader)


def validate_model(model, data, labels=None, nn=False):
    np_test_data = np.array(data,dtype=np.float64)
    test_dataset = TensorDataset(torch.tensor(np_test_data).float())
    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                                 shuffle=False, num_workers=4)
    predictor = L.Trainer(devices=1)
    predictions_all_batches = predictor.predict(model, dataloaders=test_generator)
    if not nn:
        pred_labels = [1 if z >= 0 else 0 for z in
                       [y.tolist() for x in predictions_all_batches for y in x]]
    else:
        pred_labels = [1 if z[0] >= 0 else 0 for z in
                       [y.tolist() for x in predictions_all_batches for y in x]]
    if labels==None:
        return pred_labels
    else:
        return f1_score(labels, pred_labels)