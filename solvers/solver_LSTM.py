import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import pandas as pd
import numpy as np
import os 
import random
solvers_dir = os.path.dirname(__file__)  # Current solver directory
main_dir = os.path.dirname(solvers_dir)  # Project main directory
utils_dir = os.path.join(main_dir,'utils')
sys.path.append(utils_dir)
from my_dataloader import get_loader_series # type: ignore
models_dir = os.path.join(main_dir,'models')
sys.path.append(models_dir)
from model_LSTM import * # type: ignore
from datetime import datetime
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class solver_LSTM(object):
    DEFAULTS = {}
    def __init__(self, config):
        self.__dict__.update(solver_LSTM.DEFAULTS, **config)
        self.train_dataloader = get_loader_series(
            batch_size=self.LSTM_batch_size,
            win_size=self.LSTM_win_size,
            step=self.LSTM_step,
            mode="train",
            traindata_path=self.LSTM_traindata_path,
            testdata_path=self.LSTM_testdata_path,
            subseq_length=576)
        self.vali_dataloader = get_loader_series(
            batch_size=self.LSTM_batch_size,
            win_size=self.LSTM_win_size,
            step=self.LSTM_step,
            mode="vali",
            traindata_path=self.LSTM_traindata_path,
            testdata_path=self.LSTM_testdata_path,
            subseq_length=576)
        self.test_dataloader = get_loader_series(
            batch_size=self.LSTM_batch_size,
            win_size=self.LSTM_win_size,
            step=self.LSTM_step,
            mode="test",
            traindata_path=self.LSTM_traindata_path,
            testdata_path=self.LSTM_testdata_path,
            subseq_length=576)
        self.model = LSTM_Block(input_dim=self.LSTM_in_features,hidden_dim=self.LSTM_hidden_dim,num_layers=self.LSTM_num_layers,dropout=0.1) # type: ignore
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LSTM_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,T_max=self.LSTM_num_epochs,eta_min=1e-6)
        self.criterion = nn.MSELoss()
        self.criterion_test = nn.MSELoss(reduction='none')

    def _set_test_reproducibility(self):
        random.seed(self.LSTM_seed)
        np.random.seed(self.LSTM_seed)
        torch.manual_seed(self.LSTM_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.LSTM_seed)
            torch.cuda.manual_seed_all(self.LSTM_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    def train_step(self,train_data):
        self.optimizer.zero_grad()
        output = self.model(train_data)
        loss = self.criterion(output, train_data)
        loss.backward()
        self.optimizer.step()

    def validate(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for i, (train_data, labels) in enumerate(self.vali_dataloader):
                output = self.model(train_data)
                loss = self.criterion(output, train_data)
                losses.append(loss.cpu().item())
            mean_loss = np.mean(losses)
        return mean_loss

    def train(self):
        print("------start training------")
        loss_record = []
        for epoch in range(self.LSTM_num_epochs):
            self.model.train()
            for i, (train_data, labels) in enumerate(self.train_dataloader):
                self.train_step(train_data)
            mean_loss = self.validate()
            print("Epoch: {0} MSEloss: {1:.4f}".format(epoch,mean_loss))
            loss_record.append(mean_loss)
            torch.save(self.model.state_dict(), os.path.join(self.LSTM_checkpoint_path, 'checkpoint.pth'))
        loss_record = pd.DataFrame(loss_record, columns=['MSE loss'])
        loss_record.to_excel(os.path.join(self.LSTM_checkpoint_path,'log_vali_loss.xlsx'), index=False)


    def test(self):
        print('loading the pretrained model')
        self._set_test_reproducibility()
        self.model.load_state_dict(torch.load(self.LSTM_pretrained_model_path))
        self.model.eval()
        print("------start testing------")

        all_losses = []
        all_labels = []

        with torch.no_grad():
            for i, (test_data, labels) in enumerate(self.test_dataloader):
                # test_data: (B, W, F)
                # output:    (B, W, F)
                output = self.model(test_data)

                # feature-level -> point-level
                # (B, W, F) -> (B, W)
                point_loss = torch.max(self.criterion_test(output, test_data),dim=-1).values

                all_losses.append(point_loss)
                all_labels.append(labels)  # (B, W)

        all_losses = torch.cat(all_losses, dim=0)   # (N, W)
        all_labels = torch.cat(all_labels, dim=0)   # (N, W)

        # point-level flatten
        losses_np = all_losses.reshape(-1).cpu().numpy()
        labels_np = all_labels.reshape(-1).cpu().numpy()

        threshold_percentile = 93
        threshold = np.percentile(losses_np, threshold_percentile)

        predictions = (losses_np >= threshold).astype(int)

        cm = confusion_matrix(labels_np, predictions)
        print("\n=== Confusion Matrix (Point-level) ===")
        print(f"True Negative: {cm[0, 0]}")
        print(f"False Positive: {cm[0, 1]}")
        print(f"False Negative: {cm[1, 0]}")
        print(f"True Positive: {cm[1, 1]}")

        accuracy = accuracy_score(labels_np, predictions)
        precision = precision_score(labels_np, predictions, zero_division=0)
        recall = recall_score(labels_np, predictions, zero_division=0)
        f1 = f1_score(labels_np, predictions, zero_division=0)

        print("\n=== Point-level Metrics ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return precision, recall, f1
