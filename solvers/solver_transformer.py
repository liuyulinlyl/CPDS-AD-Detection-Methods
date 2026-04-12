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
from my_dataloader import get_loader_series   # type: ignore
models_dir = os.path.join(main_dir,'models')
sys.path.append(models_dir)
from model_transformer import Transformer # type: ignore
from datetime import datetime
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class solver_transformer(object):
    DEFAULTS = {}
    def __init__(self, config):
        self.__dict__.update(solver_transformer.DEFAULTS, **config)
        # with open(self.traindata_path, 'rb') as f:  # Binary read mode
        #     train_dataframes = pickle.load(f)
        # standardized_train_dict, fitted_scaler = standardize_dfs_dict(train_dataframes) # type: ignore
        # self.train_dataloader = time_series_dataloader(dataframes_dict=standardized_train_dict,window_size=self.win_size,batch_size=self.batch_size) # type: ignore
        # self.vali_dataloader = self.train_dataloader
        self.train_dataloader = get_loader_series(
            batch_size=self.transformer_batch_size,
            win_size=self.transformer_win_size,
            step=self.transformer_step,
            mode="train",
            traindata_path=self.transformer_traindata_path,
            testdata_path=self.transformer_testdata_path,
            subseq_length=576)
        self.vali_dataloader = get_loader_series(
            batch_size=self.transformer_batch_size,
            win_size=self.transformer_win_size,
            step=self.transformer_step,
            mode="vali",
            traindata_path=self.transformer_traindata_path,
            testdata_path=self.transformer_testdata_path,
            subseq_length=576)
        self.test_dataloader = get_loader_series(
            batch_size=self.transformer_batch_size,
            win_size=self.transformer_win_size,
            step=self.transformer_step,
            mode="test",
            traindata_path=self.transformer_traindata_path,
            testdata_path=self.transformer_testdata_path,
            subseq_length=576)

        self.model = Transformer(enc_in=self.transformer_in_features, c_out=self.transformer_in_features, d_model=self.transformer_d_model, 
                n_heads=self.transformer_nheads, e_layers=self.transformer_num_layers, d_ff=self.transformer_d_model,dropout=0.0, activation='gelu')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.transformer_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,T_max=self.transformer_num_epochs,eta_min=1e-6)
        self.criterion = nn.MSELoss()
        self.criterion_test = nn.MSELoss(reduction='none')

    def _set_test_reproducibility(self):
        random.seed(self.transformer_seed)
        np.random.seed(self.transformer_seed)
        torch.manual_seed(self.transformer_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.transformer_seed)
            torch.cuda.manual_seed_all(self.transformer_seed)
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
        for epoch in range(self.transformer_num_epochs):
            self.model.train()
            for i, (train_data, labels) in enumerate(self.train_dataloader):
                self.train_step(train_data)
            mean_loss = self.validate()
            self.scheduler.step()
            print("Epoch: {0} MSEloss: {1:.4f}".format(epoch,mean_loss))
            loss_record.append(mean_loss)
            torch.save(self.model.state_dict(), os.path.join(self.transformer_checkpoint_path, 'checkpoint.pth'))
        loss_record = pd.DataFrame(loss_record, columns=['MSE loss'])
        loss_record.to_excel(os.path.join(self.transformer_checkpoint_path,'log_vali_loss.xlsx'), index=False)


    def test(self):
        print('loading the pretrained model')
        self._set_test_reproducibility()
        self.model.load_state_dict(torch.load(self.transformer_pretrained_model_path))
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
