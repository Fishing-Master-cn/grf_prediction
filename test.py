import warnings
import yaml
import os
import argparse
from net.net import *
from data_pipeline import FORCE_Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from utils.tool import make_print_to_file

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='grf_prediction_data/input_feature',
                        help='path for input features files')
    parser.add_argument('-o', '--output_dir', type=str, default='grf_prediction_data/output_feature',
                        help='path for output features files')
    parser.add_argument('-c', '--checkpoint', type=str, default='checkpoints/Lstm_checkpoint_final.pth',
                        help='checkpoint path')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    input_root = args.input_dir
    output_root = args.output_dir
    make_print_to_file()

    output_dir = './test_result'
    os.makedirs(output_dir, exist_ok=True)

    with open('./config.cfg', 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {"Neural": NeuralNetwork, "Lstm": LstmRNN}
    output_dim = cfg['TRAIN']['OUTPUT_DIM']
    input_dim = cfg['TRAIN']['INPUT_DIM']
    hidden_dim = cfg['TRAIN']['HIDDEN_DIM']
    model_name = cfg['MODEL']['TYPE']  # 模型类型

    MSE_LOSS = torch.nn.MSELoss(reduction='mean')

    model = model_dict[model_name](input_dim, hidden_dim, output_dim).to(device).eval()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    test_list = np.load(r"test_list.npy")
    test_list = [str(i) for i in test_list]
    test_set = FORCE_Dataset(input_root, output_root, test_list)
    testDataLoader = DataLoader(test_set, batch_size=1)

    prediction_list = []
    label_list = []
    mse_loss_list = []
    rmse_loss_list = []

    with torch.no_grad():

        for index, sample in enumerate(testDataLoader):
            input_data = sample['X'].to(device).float()
            label_data = sample['y'].to(device).double()

            prediction_data = model(input_data).cpu()
            label_data = label_data.cpu()

            mse_loss = MSE_LOSS(prediction_data, label_data)
            rmse_loss = torch.sqrt(MSE_LOSS(prediction_data, label_data))
            mse_loss_list.append(mse_loss)
            rmse_loss_list.append(rmse_loss)

            prediction_list.append(np.array(prediction_data))
            label_list.append(np.array(label_data))
            interval = 100

            if (index + 1) % interval == 0:
                prediction_list = np.array(prediction_list)
                label_list = np.array(label_list)
                r2 = r2_score(prediction_list[:, :, 0], label_list[:, :, 0], sample_weight=None,
                              multioutput='uniform_average')
                
                plt.figure(dpi=500)
                plt.grid(True)
                plt.plot(prediction_list[:, 0, 0], c='r')
                plt.plot(label_list[:, 0, 0])
                plt.legend(['Prediction', "True"])

                plt.savefig(r"test_result/mse_{:.4}_rmse_{:.4}_nrmse_{:.4}_r_{:.4}.png".format(np.mean(
                    mse_loss_list),
                    np.mean(
                        rmse_loss_list),
                    np.mean(
                        rmse_loss_list) / (
                            np.max(
                                label_list) - np.min(
                        label_list)),
                    r2))
                plt.show()

                print('Processed {} / {} | mse loss {} | rmse loss {} | nrmse loss {} | r {}'.format(index + 1,
                                                                                                     len(test_list),
                                                                                                     np.mean(
                                                                                                         mse_loss_list),
                                                                                                     np.mean(
                                                                                                         rmse_loss_list),
                                                                                                     np.mean(
                                                                                                         rmse_loss_list) / (
                                                                                                             np.max(
                                                                                                                 label_list) - np.min(
                                                                                                         label_list)),
                                                                                                     r2))

                prediction_list = []
                label_list = []
                mse_loss_list = []
                rmse_loss_list = []


if __name__ == '__main__':
    main()
