import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class my_dataloader_series:
    def __init__(self, win_size, step, mode="train", traindata_path='', testdata_path='', subseq_length=None):
        """
        Args:
            win_size: Sliding window size.
            step: Sliding window step.
            mode: Mode ('train', 'vali', 'test').
            traindata_path: Training data path.
            testdata_path: Test data path.
            subseq_length: Subsequence length; use None to skip subsequence splitting.
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.subseq_length = subseq_length   
        self.scaler = StandardScaler()
        print('loading the train dataset......')
        train_dataset = pd.read_excel(traindata_path, index_col=0)
        print('normalizing the train data with StandardScaler......')
        self.scaler.fit(train_dataset)
        train_dataset = self.scaler.transform(train_dataset)
        rows, columns = train_dataset.shape
        split_int = int(0.7*(rows//self.subseq_length))
        self.split_point = self.subseq_length*split_int

        # Split training data into subsequences.
        if self.subseq_length is not None:
            self.train_subseqs = self._split_into_subsequences(train_dataset[:self.split_point])
            self.vali_subseqs = self._split_into_subsequences(train_dataset[self.split_point:])
        else:
            # Use the original continuous data when no subsequence length is set.
            self.train_data = train_dataset[:self.split_point]
            self.vali_data = train_dataset[self.split_point:]

        print('loading the test dataset......')
        test_dataset = pd.read_excel(testdata_path, index_col=0)
        self.test_data = test_dataset.iloc[:, :-1]
        #self.test_data = test_dataset
        self.test_data = self.scaler.transform(self.test_data) 
        self.test_label = test_dataset.iloc[:, -1].values

        # Split test data into subsequences.
        if self.subseq_length is not None:
            self.test_subseqs = self._split_into_subsequences(self.test_data)
            # Split test labels the same way.
            self.test_label_subseqs = self._split_into_subsequences(
                self.test_label.reshape(-1, 1)
            )

    def _split_into_subsequences(self, data):
        """
        Split data into subsequences of length subseq_length.
        Drop the final subsequence if it is too short.
        """
        if self.subseq_length is None:
            return [data]
        
        subseqs = []
        n_samples = len(data)
        n_subseqs = n_samples // self.subseq_length
        
        for i in range(n_subseqs):
            start_idx = i * self.subseq_length
            end_idx = start_idx + self.subseq_length
            subseq = data[start_idx:end_idx]
            if len(subseq) == self.subseq_length:  # Ensure the length is correct.
                subseqs.append(subseq)
        
        return subseqs


    def __len__(self):
        if self.subseq_length is not None:
            total_windows = 0
            if self.mode == "train":
                for subseq in self.train_subseqs:
                    # Number of windows inside each subsequence.
                    n_windows = (len(subseq) - self.win_size) // self.step + 1
                    if n_windows > 0:
                        total_windows += n_windows
            elif self.mode == 'vali':
                for subseq in self.vali_subseqs:
                    n_windows = (len(subseq) - self.win_size) // self.step + 1
                    if n_windows > 0:
                        total_windows += n_windows
            elif self.mode == 'test':
                for subseq in self.test_subseqs:
                    n_windows = (len(subseq) - self.win_size) // self.step + 1
                    if n_windows > 0:
                        total_windows += n_windows
            return total_windows
        else:
            # Original continuous mode.
            if self.mode == "train":
                return (len(self.train_data) - self.win_size) // self.step + 1
            elif self.mode == 'vali':
                return (len(self.vali_data) - self.win_size) // self.step + 1
            elif self.mode == 'test':
                return (len(self.test_data) - self.win_size) // self.step + 1
        
    def __getitem__(self, index):
        if self.subseq_length is None:
            # Use the original continuous mode.
            index = index * self.step
            if self.mode == "train":
                return (np.float32(self.train_data[index:index + self.win_size]), 
                        np.float32(self.test_label[0:self.win_size]))
            elif self.mode == 'vali':
                return (np.float32(self.vali_data[index:index + self.win_size]), 
                        np.float32(self.test_label[0:self.win_size]))
            elif self.mode == 'test':
                return (np.float32(self.test_data[index:index + self.win_size]), 
                        np.float32(self.test_label[index:index + self.win_size]))
        
        # Find the matching subsequence and local position.
        if self.mode == "train":
            subseqs = self.train_subseqs
            labels = None  # Training mode has no labels.
        elif self.mode == 'vali':
            subseqs = self.vali_subseqs
            labels = None
        elif self.mode == 'test':
            subseqs = self.test_subseqs
            label_subseqs = self.test_label_subseqs
        
        # Walk through subsequences to locate the requested index.
        current_idx = 0
        for subseq_idx, subseq in enumerate(subseqs):
            # Number of windows in the current subsequence.
            n_windows = (len(subseq) - self.win_size) // self.step + 1
            if n_windows <= 0:
                continue
            
            if index < current_idx + n_windows:
                # The index falls in the current subsequence.
                local_idx = index - current_idx  # Window index within the subsequence.
                start_pos = local_idx * self.step
                end_pos = start_pos + self.win_size
                
                # Get data.
                data_window = subseq[start_pos:end_pos]
                
                # Get labels in test mode.
                if self.mode == 'test':
                    label_window = label_subseqs[subseq_idx][start_pos:end_pos]
                    return np.float32(data_window), np.float32(label_window.flatten())
                else:
                    # Return dummy labels for train and validation modes.
                    dummy_label = np.zeros(self.win_size, dtype=np.float32)
                    return np.float32(data_window), dummy_label
            
            current_idx += n_windows
        
        raise IndexError(f"Index {index} out of range")

def get_loader_series(batch_size, win_size, step, mode="train", 
                      traindata_path='', testdata_path='', subseq_length=None):
    """
    Create a data loader.
    
    Args:
        subseq_length: Subsequence length; use None for the original continuous mode.
    """
    dataset = my_dataloader_series(win_size, step, mode, traindata_path, 
                                   testdata_path, subseq_length)
    shuffle = False
    if mode == 'train':
        shuffle = True
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False  # Keep incomplete final batches.
    )
    return data_loader

if __name__ == "__main__":
    # Use subsequence mode.
    test_loader = get_loader_series(
        batch_size=32,
        win_size=100,
        step=1,
        mode="test",
        traindata_path="train_data.xlsx",
        testdata_path="test_data_FDIA.xlsx",
        subseq_length=576  # Length of each subsequence.
    )
    for i, (input_data, labels) in enumerate(test_loader):
        print(labels.shape)


class my_dataloader:
    def __init__(self,mode="train",traindata_path='',testdata_path=''):
        self.mode = mode 
        self.scaler = StandardScaler()
        print('loading the train dataset......')
        train_dataset = pd.read_excel(traindata_path, index_col=0)
        print('normalizing the train data with StandardScaler......')
        self.scaler.fit(train_dataset)
        train_dataset = self.scaler.transform(train_dataset)
        split_point = int(len(train_dataset) * 0.7)
        self.train_data = train_dataset[:split_point]  # First 70%.
        self.vali_data = train_dataset[split_point:]  # Remaining 30%.

        print('loading the test dataset......')
        test_dataset = pd.read_excel(testdata_path, index_col=0)
        self.test_data = test_dataset.iloc[:, :-1]
        #self.test_data = test_dataset
        self.test_data = self.scaler.transform(self.test_data) 
        self.test_label = test_dataset.iloc[:,-1]

    def __len__(self):
        if self.mode == "train":
            return self.train_data.shape[0]
        elif (self.mode == 'vali'):
            return self.vali_data.shape[0]
        elif (self.mode == 'test'):
            return self.test_data.shape[0]
        
    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train_data[index]), np.float32(self.test_label[0])
        elif (self.mode == 'vali'):
            return np.float32(self.vali_data[index]), np.float32(self.test_label[0])
        elif (self.mode == 'test'):
            return np.float32(self.test_data[index]), np.float32(self.test_label[index])


def get_loader(batch_size,mode="train",traindata_path='',testdata_path=''):
    dataset = my_dataloader(mode,traindata_path,testdata_path)
    shuffle = False
    if mode == 'train':
        shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader


