import os, sys
import torch
import numpy as np
import argparse
from argparse import Namespace
import tqdm
import six
from scipy import stats
from helper import set_seeds
from torch.utils.data import DataLoader
from utils.datasets import get_scaled_data
from utils.q_model_ens import QModelEns, MSEModel
from losses import batch_qr_loss, batch_interval_loss, RQR_loss
import helper
from helper import REAL_DATA
import warnings
warnings.filterwarnings("ignore")
sys.modules['sklearn.externals.six'] = six

os.environ["MKL_CBWR"] = 'AUTO'

results_path = helper.results_path

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def get_loss_fn(loss_name):
    if loss_name == 'batch_qr':
        fn = batch_qr_loss
    elif loss_name == 'batch_int':
        fn = batch_interval_loss
    elif loss_name == 'RQR':
        fn = RQR_loss
    else:
        raise ValueError('loss arg not valid')

    return fn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')

    parser.add_argument('--seed_begin', type=int, default=None,
                        help='random seed')
    parser.add_argument('--seed_end', type=int, default=None,
                        help='random seed')

    parser.add_argument('--data', type=str, default='',
                        help='dataset to use')

    parser.add_argument('--num_q', type=int, default=30,
                        help='number of quantiles you want to sample each step')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu num to use')

    parser.add_argument('--num_ep', type=int, default=10000,
                        help='number of epochs')
    parser.add_argument('--nl', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--hs', type=int, default=64,
                        help='hidden size')

    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout ratio of the dropout level')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--bs', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--wait', type=int, default=200,
                        help='how long to wait for lower validation loss')

    parser.add_argument('--loss', type=str,
                        help='specify type of loss')

    parser.add_argument('--corr_mult', type=float, default=0.,
                        help='correlation penalty multiplier')

    parser.add_argument('--hsic_mult', type=float, default=0.,
                        help='correlation penalty multiplier')

    parser.add_argument('--test_ratio', type=float, default=0.4,
                        help='ratio of test set size')

    parser.add_argument('--save_training_results', type=int, default=0,
                        help='1 for saving results during training, or 0 for not saving')

    parser.add_argument('--method', type=str, default='QR',
                        help='method to use (QR or qr_forest)')
    
    parser.add_argument('--penalty', type=float, default=0,
                    help='penalty coefficient for PI loss')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args.device = device
    if args.method not in ['QR', 'qr_forest']:
        raise ValueError('method arg not valid')

    return args


def update_results_during_training(model, x, y, set_name, results_dict, alpha):
    with torch.no_grad():
        if len(x) == 0 or len(y) == 0:
            return
        y = y.reshape(-1).to(device)
        idx = np.random.permutation(len(x))  # [:len(xx)]
        x = x[idx].to(device)
        quantiles = torch.Tensor([alpha / 2, 1 - alpha / 2]).to(device)
        test_preds = model.predict_q(
            x, quantiles, ens_pred_type='conf',
            recal_model=None, recal_type=None, use_best_va_model=False
        )
        test_preds.detach().cpu().numpy()
        y_upper = test_preds[:, 1].detach().cpu().numpy()
        y_lower = test_preds[:, 0].detach().cpu().numpy()

        if torch.is_tensor(y):
            curr_y = y.cpu().detach().numpy()[idx]
        else:
            curr_y = y[idx]
        in_the_range = ((curr_y >= y_lower) & (curr_y <= y_upper))
        lengths = (y_upper - y_lower)

        if 'pearsons_correlation' + '_over_' + set_name not in results_dict:
            results_dict['pearsons_correlation' + '_over_' + set_name] = []

        results_dict['pearsons_correlation' + '_over_' + set_name] += [
            stats.pearsonr(in_the_range, lengths)[0]]

        if 'coverage' + '_over_' + set_name not in results_dict:
            results_dict['coverage' + '_over_' + set_name] = []

        results_dict['coverage' + '_over_' + set_name] += [np.mean(in_the_range)]

        if 'interval_lengths' + '_over_' + set_name not in results_dict:
            results_dict['interval_lengths' + '_over_' + set_name] = []

        results_dict['interval_lengths' + '_over_' + set_name] += [np.mean(lengths)]


if __name__ == '__main__':

    args = parse_args()
    args.num_ens = 1
    args.boot = 0

    POSSIBLE_REAL_DATA_NAMES = ['kin8nm', 'naval', 'boston', 'energy', 'power', 'protein', 'wine', 'yacht', 'concrete']

    data_type = REAL_DATA

    if data_type == REAL_DATA:

        if args.data != '' and args.data.replace("scaled_", '') not in POSSIBLE_REAL_DATA_NAMES:
            raise RuntimeError('Must choose possible data name!')

        if args.data.replace("scaled_", '') in POSSIBLE_REAL_DATA_NAMES:
            REAL_DATA_NAMES = [args.data]
        else:
            REAL_DATA_NAMES = []
        DATA_NAMES = REAL_DATA_NAMES
    else:
        assert data_type == SYN_DATA
        if str(args.data) in [str(i) for i in range(0, 11)]:
            SYN_DATA_NAMES = [args.data]
        else:
            SYN_DATA_NAMES = ['3', '10']
        DATA_NAMES = SYN_DATA_NAMES

    if args.seed is not None:
        SEEDS = [args.seed]
    elif args.seed_begin is not None and args.seed_end is not None:
        SEEDS = range(args.seed_begin, args.seed_end)
    else:
        SEEDS = range(0, 30)

    save_results_during_training = False #bool(args.save_training_results)

    # print('DEVICE: {}'.format(args.device))

    alpha = 0.1

    args_summary = helper.args_to_txt(args)

    if 'int' in args.loss:
        TRAINING_OVER_ALL_QUANTILES = True
        # print("Training over all qunatiles")
    elif 'qr' in args.loss:
        TRAINING_OVER_ALL_QUANTILES = True
        # print("Training over two qunatiles")
    elif 'pi' in args.loss:
        TRAINING_OVER_ALL_QUANTILES = False
        # print("PI loss")
    else:
        assert False

    for d in DATA_NAMES:
        if save_results_during_training:
            results_during_training = {}
            for s in SEEDS:
                results_during_training[s] = {}

        for s in tqdm.tqdm(SEEDS):

            args.data = str(d)
            args.seed = s

            set_seeds(args.seed)
            data_args = Namespace(dataset=args.data, seed=args.seed)

            # Fetching data
            data_out = get_scaled_data(args)
            x_train, y_train = data_out.x_train, data_out.y_train
            unscaled_x_train = None
            unscaled_x_test = None
            minority_group_uncertainty = None
            group_feature = None

            
            x_tr, x_va, x_te, y_tr, y_va, y_te, y_al = \
                data_out.x_tr, data_out.x_va, data_out.x_te, data_out.y_tr, \
                data_out.y_va, data_out.y_te, data_out.y_al
            x_va, y_va = x_va.to(args.device), y_va.to(args.device)
            x_tr, y_tr = x_tr.to(args.device), y_tr.to(args.device)
            y_te, x_te = y_te.to(args.device), x_te.to(args.device)
            x_train, y_train = x_train.to(args.device), y_train.to(args.device)
            x_test = x_te
            y_test = y_te

            w_tr, w_va, get_tr_weights = helper.get_wqr_weights(args.loss, x_tr, y_tr, x_va, y_va, args)


            y_range = (y_al.max() - y_al.min()).item()

            # creating the model
            num_tr = x_tr.shape[0]
            dim_x = x_tr.shape[1]
            dim_y = y_tr.shape[1]
            # print("Hidden size : ", args.hs)
            # print("Learning rate : ", args.lr)
            # print("Penalty : ", args.penalty)
            # print("Dropout : ", args.dropout)
            # print("Batch size : ", args.bs)
            # print("Weight decay : ", args.wd)
            # print("epochs : ", args.num_ep)
            # print('DEVICE: {}'.format(args.device))

            if args.loss == 'RQR':
                model_ens = QModelEns(input_size=dim_x + 1, output_size=2,
                                     hidden_size=args.hs, num_layers=args.nl, dropout=args.dropout,
                                     lr=args.lr, wd=args.wd,
                                     num_ens=args.num_ens, device=args.device)
            else:
                model_ens = QModelEns(input_size=dim_x + 1, output_size=dim_y,
                                    hidden_size=args.hs, num_layers=args.nl, dropout=args.dropout,
                                    lr=args.lr, wd=args.wd,
                                    num_ens=args.num_ens, device=args.device)

            loader = DataLoader(helper.IndexedDataset(x_tr, y_tr),
                                shuffle=True,
                                batch_size=args.bs)

            # Loss function
            loss_fn = get_loss_fn(args.loss)
            batch_loss = True if 'batch' in args.loss else False
            """ train loop """
            tr_loss_list = []
            va_loss_list = []
            te_loss_list = []
            batch_size = args.bs
            for ep in range(args.num_ep):

                if model_ens.done_training:
                    print('Done training ens at EP {}'.format(ep))
                    break

                # Take train step
                ep_train_loss = []  # list of losses from each batch, for one epoch
                epoch_loss = []

                for xi, yi, index in loader:
                    if TRAINING_OVER_ALL_QUANTILES:
                        # print("Training over all qunatiles")
                        q_list = torch.rand(args.num_q)
                    else:
                        # print("Training over two qunatiles")
                        q_list = torch.Tensor([alpha / 2])
                    loss = model_ens.loss(loss_fn, xi, yi, q_list,
                                          batch_q=batch_loss,
                                          take_step=True, args=args, weights=get_tr_weights(index))
                    ep_train_loss.append(loss)

                ep_tr_loss = np.nanmean(np.stack(ep_train_loss, axis=0), axis=0)
                tr_loss_list.append(ep_tr_loss)

                # Validation loss
                # x_va, y_va = x_va.to(args.device), y_va.to(args.device)
                if TRAINING_OVER_ALL_QUANTILES:
                    # print("Training over all qunatiles")
                    va_te_q_list = torch.linspace(0.01, 0.99, 99)
                else:
                    va_te_q_list = torch.Tensor([alpha / 2, 1 - alpha / 2])
                ep_va_loss = model_ens.update_va_loss(
                    loss_fn, x_va, y_va, va_te_q_list,
                    batch_q=batch_loss, curr_ep=ep, num_wait=args.wait,
                    args=args, weights=w_va
                )
                va_loss_list.append(ep_va_loss)

                if save_results_during_training:
                    for (x, y, set_name) in [(x_tr, y_tr, 'train'), (x_va, y_va, 'validation'),
                                                (x_te, y_te, 'test')]:
                        update_results_during_training(model_ens, x, y, set_name,
                                                        results_during_training[s], alpha)

                # Printing some losses
                # if (ep % 100 == 0) or (ep == args.num_ep - 1):
                #     print('EP:{}'.format(ep))
                #     pass
            # Move everything to cpu
            x_tr, y_tr, x_va, y_va, x_te, y_te = \
                x_tr.cpu(), y_tr.cpu(), x_va.cpu(), y_va.cpu(), x_te.cpu(), y_te.cpu()
            model_ens.use_device(torch.device('cpu'))

            quantiles = torch.Tensor([alpha / 2, 1 - alpha / 2])
            
            # bins = 10
            # quantiles_l = torch.linspace(0, alpha, bins)
            # quantiles_u = 1 - alpha + quantiles_l
            # quantiles = torch.cat([quantiles_l, quantiles_u], dim=0)
            
            test_preds = model_ens.predict_q(
                x_te, quantiles, ens_pred_type='conf',
                recal_model=None, recal_type=None
            )
            
            test_preds = test_preds.detach().cpu().numpy()


            train_preds = model_ens.predict_q(
                x_train, quantiles, ens_pred_type='conf',
                recal_model=None, recal_type=None
            )
            train_preds.detach().cpu().numpy()

            train_y_upper = train_preds[:, 1]
            train_y_lower = train_preds[:, 0]
            y_train = y_train.reshape(len(y_train))
            y_upper = test_preds[:, 1]
            y_lower = test_preds[:, 0]
            y_test = y_test.reshape(len(y_test))

            helper.save_results(d, data_type, x_train, unscaled_x_train, x_test, unscaled_x_test, y_train, y_test,
                                y_upper,
                                y_lower,
                                train_y_upper,
                                train_y_lower,
                                s,
                                args,
                                minority_group_uncertainty=minority_group_uncertainty,
                                group_feature=group_feature, y_range=y_range)

  