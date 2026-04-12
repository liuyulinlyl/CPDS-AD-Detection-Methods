import os
import argparse
import sys
from torch.backends import cudnn
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

main_dir = os.path.dirname(__file__)
solvers_dir = os.path.join(main_dir,'solvers')
sys.path.append(solvers_dir)

from solver_transformer import * # type: ignore
from solver_LSTM import * # type: ignore
from solver_TCN import * # type: ignore

main_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(main_dir,'CPDS-AD_dataset')
def main(config):
    cudnn.benchmark = True
    if config.solver == 'test_all':
        # Additive attack
        config.transformer_testdata_path = os.path.join(dataset_dir,'test_data_A.xlsx')
        solver = solver_transformer(vars(config)) # type: ignore
        A_transformer_precision, A_transformer_recall, A_transformer_f1 = solver.test()
        config.LSTM_testdata_path = os.path.join(dataset_dir,'test_data_A.xlsx')
        solver = solver_LSTM(vars(config)) # type: ignore
        A_LSTM_precision, A_LSTM_recall, A_LSTM_f1 = solver.test()
        config.TCN_testdata_path = os.path.join(dataset_dir,'test_data_A.xlsx')
        solver = solver_TCN(vars(config)) # type: ignore
        A_TCN_precision, A_TCN_recall, A_TCN_f1 = solver.test()

        # Subtractive attack
        config.transformer_testdata_path = os.path.join(dataset_dir,'test_data_S.xlsx')
        solver = solver_transformer(vars(config)) # type: ignore
        S_transformer_precision, S_transformer_recall, S_transformer_f1 = solver.test()
        config.LSTM_testdata_path = os.path.join(dataset_dir,'test_data_S.xlsx')
        solver = solver_LSTM(vars(config)) # type: ignore
        S_LSTM_precision, S_LSTM_recall, S_LSTM_f1 = solver.test()
        config.TCN_testdata_path = os.path.join(dataset_dir,'test_data_S.xlsx')
        solver = solver_TCN(vars(config)) # type: ignore
        S_TCN_precision, S_TCN_recall, S_TCN_f1 = solver.test()

        # Replay attack
        config.transformer_testdata_path = os.path.join(dataset_dir,'test_data_R.xlsx')
        solver = solver_transformer(vars(config)) # type: ignore
        R_transformer_precision, R_transformer_recall, R_transformer_f1 = solver.test()
        config.LSTM_testdata_path = os.path.join(dataset_dir,'test_data_R.xlsx')
        solver = solver_LSTM(vars(config)) # type: ignore
        R_LSTM_precision, R_LSTM_recall, R_LSTM_f1 = solver.test()
        config.TCN_testdata_path = os.path.join(dataset_dir,'test_data_R.xlsx')
        solver = solver_TCN(vars(config)) # type: ignore
        R_TCN_precision, R_TCN_recall, R_TCN_f1 = solver.test()

        data = {
            'Model': ['Transformer', 'LSTM', 'TCN'],
            'Pre(A)': [
                A_transformer_precision, A_LSTM_precision, A_TCN_precision],
            'Recall(A)': [
                A_transformer_recall, A_LSTM_recall, A_TCN_recall],
            'F1(A)': [
                A_transformer_f1, A_LSTM_f1, A_TCN_f1],
            'Pre(S)': [
                S_transformer_precision, S_LSTM_precision, S_TCN_precision],
            'Recall(S)': [
                S_transformer_recall, S_LSTM_recall, S_TCN_recall],
            'F1(S)': [
                S_transformer_f1, S_LSTM_f1, S_TCN_f1],
            'Pre(R)': [
                R_transformer_precision, R_LSTM_precision, R_TCN_precision],
            'Recall(R)': [
                R_transformer_recall, R_LSTM_recall, R_TCN_recall],
            'F1(R)': [
                R_transformer_f1, R_LSTM_f1, R_TCN_f1]
        }

        # Convert the dictionary into a DataFrame
        df = pd.DataFrame(data)

        # Specify the file path where the Excel file will be saved
        file_path = 'FDI_detection_performances.xlsx'

        # Save the DataFrame to an Excel file
        df.to_excel(file_path, index=False)

        print(f"Results saved to {file_path}")


    if config.solver == 'solver_transformer':
        solver = solver_transformer(vars(config)) # type: ignore
        if config.transformer_mode == 'train':
            solver.train()
        if config.transformer_mode == 'test':
            solver.test()
    if config.solver == 'solver_LSTM':
        solver = solver_LSTM(vars(config)) # type: ignore
        if config.LSTM_mode == 'train':
            solver.train()
        if config.LSTM_mode == 'test':
            solver.test()
    if config.solver == 'solver_TCN':
        solver = solver_TCN(vars(config)) # type: ignore
        if config.TCN_mode == 'train':
            solver.train()
        if config.TCN_mode == 'test':
            solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set the solver to run and choose different argument inputs based on it
    solver = "test_all"
    parser.add_argument('--solver', type=str, default=solver)

    parser.add_argument('--transformer_mode', type=str, default="test",choices=["train", "test"])
    # Basic training parameters
    parser.add_argument('--transformer_batch_size', type=int, default=32)
    parser.add_argument('--transformer_win_size', type=int, default=100)
    parser.add_argument('--transformer_step', type=int, default=1) # Time-series data step size
    parser.add_argument('--transformer_lr', type=float, default=0.001)
    parser.add_argument('--transformer_num_epochs', type=int, default=30)
    parser.add_argument('--transformer_seed', type=int, default=42)
    # Model architecture
    parser.add_argument('--transformer_in_features', type=int, default=132) 
    parser.add_argument('--transformer_d_model', type=int, default=256) 
    parser.add_argument('--transformer_nheads', type=int, default=8)
    parser.add_argument('--transformer_num_layers', type=int, default=1)
    # Paths
    parser.add_argument('--transformer_checkpoint_path', type=str, default=os.path.join(main_dir,'checkpoints','transformer'))
    parser.add_argument('--transformer_traindata_path', type=str, default=os.path.join(dataset_dir,'train_data.xlsx'))
    parser.add_argument('--transformer_testdata_path', type=str, default=os.path.join(dataset_dir,'test_data_R.xlsx'))
    parser.add_argument('--transformer_pretrained_model_path', type=str, default=os.path.join(main_dir,'checkpoints','transformer','checkpoint.pth'))

    parser.add_argument('--LSTM_mode', type=str, default="test",choices=["train", "test"])
    # Basic training parameters
    parser.add_argument('--LSTM_batch_size', type=int, default=32)
    parser.add_argument('--LSTM_win_size', type=int, default=30)
    parser.add_argument('--LSTM_step', type=int, default=1) # Time-series data step size
    parser.add_argument('--LSTM_lr', type=float, default=0.001)
    parser.add_argument('--LSTM_num_epochs', type=int, default=20)
    parser.add_argument('--LSTM_seed', type=int, default=42)
    # Model architecture
    parser.add_argument('--LSTM_in_features', type=int, default=132) 
    parser.add_argument('--LSTM_hidden_dim', type=int, default=256) 
    parser.add_argument('--LSTM_num_layers', type=int, default=1)
    # Paths
    parser.add_argument('--LSTM_checkpoint_path', type=str, default=os.path.join(main_dir,'checkpoints','LSTM'))
    parser.add_argument('--LSTM_traindata_path', type=str, default=os.path.join(dataset_dir,'train_data.xlsx'))
    parser.add_argument('--LSTM_testdata_path', type=str, default=os.path.join(dataset_dir,'test_data_R.xlsx'))
    parser.add_argument('--LSTM_pretrained_model_path', type=str, default=os.path.join(main_dir,'checkpoints','LSTM','checkpoint.pth'))

    parser.add_argument('--TCN_mode', type=str, default="test",choices=["train", "test"])
    # Basic training parameters
    parser.add_argument('--TCN_batch_size', type=int, default=32)
    parser.add_argument('--TCN_win_size', type=int, default=100)
    parser.add_argument('--TCN_step', type=int, default=1) # Time-series data step size
    parser.add_argument('--TCN_lr', type=float, default=0.001)
    parser.add_argument('--TCN_num_epochs', type=int, default=100)
    parser.add_argument('--TCN_seed', type=int, default=42)
    # Model architecture
    parser.add_argument('--TCN_in_features', type=int, default=132) 
    # Paths
    parser.add_argument('--TCN_checkpoint_path', type=str, default=os.path.join(main_dir,'checkpoints','TCN'))
    parser.add_argument('--TCN_traindata_path', type=str, default=os.path.join(dataset_dir,'train_data.xlsx'))
    parser.add_argument('--TCN_testdata_path', type=str, default=os.path.join(dataset_dir,'test_data_S.xlsx'))
    parser.add_argument('--TCN_pretrained_model_path', type=str, default=os.path.join(main_dir,'checkpoints','TCN','checkpoint.pth'))
        
    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    # Print all command-line arguments and their values
    main(config)
