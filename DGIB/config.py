import argparse
import torch
import os


parser = argparse.ArgumentParser()

# 1. dataset
parser.add_argument('--dataset', type=str, default='collab', help='collab, yelp, act')
parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
parser.add_argument('--nfeat', type=int, default=32, help='dim of input feature')


# 2. experiments
parser.add_argument('--mode', type=str, default='train', help='train, eval')
parser.add_argument('--attack', type=str, default='random', help='random, evasive, poisoning')
parser.add_argument('--use_cfg', type=int, default=1, help='if use configs')
parser.add_argument('--max_epoch', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--testlength', type=int, default=3, help='length for test')
parser.add_argument('--device', type=str, default='gpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models')
parser.add_argument('--split', type=int, default=0, help='dataset split')
parser.add_argument('--warm_epoch', type=int, default=0, help='the number of warm epoches')
parser.add_argument('--output_folder', type=str, default='', help='need to be modified')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')
parser.add_argument('--min_epoch', type=int, default=50, help='min epoch')
parser.add_argument('--log_dir', type=str, default="../logs/")
parser.add_argument('--log_interval', type=int, default=10, help='every n epoches to log')
parser.add_argument('--nhid', type=int, default=128, help='dim of hidden embedding')
parser.add_argument('--n_layers', type=int, default=1, help='number of hidden layers')
parser.add_argument('--heads', type=int, default=1, help='attention heads')
parser.add_argument('--norm', type=int, default=1, help='normalization')
parser.add_argument('--nbsz', type=int, default=20, help='number of sampling neighbors')
parser.add_argument('--sample_size', type=int, default=50, help='how many Z to sample for each feature X')
parser.add_argument('--maxiter', type=int, default=4, help='number of iteration')
parser.add_argument('--skip', type=int, default=0, help='')
parser.add_argument('--dropout', type=float, default=0.01, help='dropout rate')
parser.add_argument('--use_RTE', type=int, default=1, help='Relative Time Encoding')
parser.add_argument('--agg_param', type=float, default=0.15, help='aggregation weights')
parser.add_argument('--reparam_mode', type=str, default='diag', help='Reparameterization mode for XIB. Choose from "None", "diag" or "full"')
parser.add_argument('--prior_mode', type=str, default='Gaussian', help='Prior mode. Choose from "Gaussian" or "mixGau-100" (mixture of 100 Gaussian components)')
parser.add_argument('--distribution', type=str, default='Bernoulli', help='categorical,Bernoulli')
parser.add_argument('--temperature', type=float, default='0.2', help='temperature')
parser.add_argument('--alpha', type=float, default=0.1, help='hyperparameter 1')
parser.add_argument('--beta', type=float, default=0.1, help='hyperparameter 2')

args = parser.parse_args()


# 3. set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda:{}".format(args.device_id))
    print('using gpu:{} to train the model'.format(args.device_id))
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')


def setargs(args, hp):
    for k, v in hp.items():
        setattr(args, k, v)


if args.use_cfg:
    if args.distribution == 'Bernoulli':
        # Bernoulli
        if 'collab' in args.dataset:
            hp = {"distribution": "Bernoulli", "lr": 1e-05, "nbsz": 50, "sample_size": 100, "dropout": 1e-05, "temperature": 10.0, "alpha": 0.1, "beta": 0.1}
            setargs(args, hp)
        elif 'yelp' in args.dataset:
            hp = {"distribution": "Bernoulli", "lr": 1e-06, "nbsz": 15, "sample_size": 130, "dropout": 1e-05, "temperature": 10.0, "alpha": 0.5, "beta": 0.5}
            setargs(args, hp)
        elif 'act' in args.dataset:
            hp = {"distribution": "Bernoulli", "lr": 1e-06, "nbsz": 30, "sample_size": 60, "dropout": 1e-05, "temperature": 10.0, "alpha": 0.5, "beta": 0.5}
            setargs(args, hp)
    
    elif args.distribution == 'categorical':
        # Categorical
        if 'collab' in args.dataset:
            hp = {"distribution": "categorical", "lr": 0.0001, "nbsz": 20, "sample_size": 50, "dropout": 0.01, "temperature": 0.2, "alpha": 0.1, "beta": 0.1}
            setargs(args, hp)
        elif 'yelp' in args.dataset:
            hp = {"distribution": "categorical", "lr": 0.001, "nbsz": 50, "sample_size": 50, "dropout": 0.01, "temperature": 0.2, "alpha": 0.01, "beta": 0.03}
            setargs(args, hp)
        elif 'act' in args.dataset:
            hp = {"distribution": "categorical", "lr": 0.001, "nbsz": 50, "sample_size": 50, "dropout": 0.01, "temperature": 0.2, "alpha": 0.1, "beta": 0.1}
            setargs(args, hp)

    else:
        raise NotImplementedError(f"dataset {args.dataset} not implemented")