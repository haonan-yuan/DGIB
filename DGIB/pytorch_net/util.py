from __future__ import print_function
from collections import Counter, OrderedDict
import os
from numbers import Number
import numpy as np
from copy import deepcopy
import itertools
import json
import operator
import pickle
import random
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
import scipy.linalg
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler

PrecisionFloorLoss = 2 ** (-32)
CLASS_TYPES = [
    "MLP",
    "Multi_MLP",
    "Branching_Net",
    "Fan_in_MLP",
    "Model_Ensemble",
    "Model_with_uncertainty",
    "RNNCellBase",
    "LSTM",
    "Wide_ResNet",
    "Conv_Net",
    "Conv_Model",
    "Conv_Autoencoder",
    "VAE",
    "Net_reparam",
    "Mixture_Gaussian",
    "Triangular_dist",
]
ACTIVATION_LIST = [
    "relu",
    "leakyRelu",
    "leakyReluFlat",
    "tanh",
    "softplus",
    "sigmoid",
    "selu",
    "elu",
    "sign",
    "heaviside",
    "softmax",
    "negLogSoftmax",
    "naturalLogSoftmax",
]


def plot_matrices(
    matrix_list,
    shape=None,
    images_per_row=10,
    scale_limit=None,
    figsize=(20, 8),
    x_axis_list=None,
    filename=None,
    title=None,
    subtitles=[],
    highlight_bad_values=True,
    plt=None,
    pdf=None,
):
    """Plot the images for each matrix in the matrix_list."""
    import matplotlib
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=figsize)
    fig.set_canvas(plt.gcf().canvas)
    if title is not None:
        fig.suptitle(title, fontsize=18, horizontalalignment="left", x=0.1)

    # To np array. If None, will transform to NaN:
    matrix_list_new = []
    for i, element in enumerate(matrix_list):
        if element is not None:
            matrix_list_new.append(to_np_array(element))
        else:
            matrix_list_new.append(np.array([[np.NaN]]))
    matrix_list = matrix_list_new

    num_matrixs = len(matrix_list)
    rows = np.ceil(num_matrixs / float(images_per_row))
    try:
        matrix_list_reshaped = (
            np.reshape(np.array(matrix_list), (-1, shape[0], shape[1]))
            if shape is not None
            else np.array(matrix_list)
        )
    except:
        matrix_list_reshaped = matrix_list
    if scale_limit == "auto":
        scale_min = np.Inf
        scale_max = -np.Inf
        for matrix in matrix_list:
            scale_min = min(scale_min, np.min(matrix))
            scale_max = max(scale_max, np.max(matrix))
        scale_limit = (scale_min, scale_max)
    for i in range(len(matrix_list)):
        ax = fig.add_subplot(rows, images_per_row, i + 1)
        image = matrix_list_reshaped[i].astype(float)
        if len(image.shape) == 1:
            image = np.expand_dims(image, 1)
        if highlight_bad_values:
            cmap = matplotlib.cm.binary
            cmap.set_bad("red", alpha=0.2)
            mask_key = []
            mask_key.append(np.isnan(image))
            mask_key.append(np.isinf(image))
            mask_key = np.any(np.array(mask_key), axis=0)
            image = np.ma.array(image, mask=mask_key)
        else:
            cmap = matplotlib.cm.binary
        if scale_limit is None:
            ax.matshow(image, cmap=cmap)
        else:
            assert len(scale_limit) == 2, "scale_limit should be a 2-tuple!"
            ax.matshow(image, cmap=cmap, vmin=scale_limit[0], vmax=scale_limit[1])
        if len(subtitles) > 0:
            ax.set_title(subtitles[i])
        try:
            xlabel = "({0:.4f},{1:.4f})\nshape: ({2}, {3})".format(
                np.min(image), np.max(image), image.shape[0], image.shape[1]
            )
            if x_axis_list is not None:
                xlabel += "\n{}".format(x_axis_list[i])
            plt.xlabel(xlabel)
        except:
            pass
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    else:
        plt.show()

    if scale_limit is not None:
        print("scale_limit: ({0:.6f}, {1:.6f})".format(scale_limit[0], scale_limit[1]))
    print()


class Recursive_Loader(object):
    """A recursive loader, able to deal with any depth of X"""

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.length = int(len(self.y) / self.batch_size)
        self.idx_list = torch.randperm(len(self.y))

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < self.length:
            idx = self.idx_list[
                self.current * self.batch_size : (self.current + 1) * self.batch_size
            ]
            self.current += 1
            return recursive_index((self.X, self.y), idx)
        else:
            self.idx_list = torch.randperm(len(self.y))
            raise StopIteration


def recursive_index(data, idx):
    """Recursively obtain the idx of data"""
    data_new = []
    for i, element in enumerate(data):
        if isinstance(element, tuple):
            data_new.append(recursive_index(element, idx))
        else:
            data_new.append(element[idx])
    return data_new


def record_data(
    data_record_dict, data_list, key_list, nolist=False, ignore_duplicate=False
):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(key_list, list):
        key_list = [key_list]
    assert len(data_list) == len(
        key_list
    ), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        if nolist:
            data_record_dict[key] = data
        else:
            if key not in data_record_dict:
                data_record_dict[key] = [data]
            else:
                if (not ignore_duplicate) or (data not in data_record_dict[key]):
                    data_record_dict[key].append(data)


def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if (
            isinstance(array, torch.Tensor)
            or isinstance(array, torch.FloatTensor)
            or isinstance(array, torch.LongTensor)
            or isinstance(array, torch.ByteTensor)
            or isinstance(array, torch.cuda.FloatTensor)
            or isinstance(array, torch.cuda.LongTensor)
            or isinstance(array, torch.cuda.ByteTensor)
        ):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def to_Variable(*arrays, **kwargs):
    """Transform numpy arrays into torch tensors/Variables"""
    is_cuda = kwargs["is_cuda"] if "is_cuda" in kwargs else False
    requires_grad = kwargs["requires_grad"] if "requires_grad" in kwargs else False
    array_list = []
    for array in arrays:
        is_int = False
        if isinstance(array, Number):
            is_int = True if isinstance(array, int) else False
            array = [array]
        if (
            isinstance(array, np.ndarray)
            or isinstance(array, list)
            or isinstance(array, tuple)
        ):
            is_int = True if np.array(array).dtype.name == "int64" else False
            array = torch.tensor(array).float()
        if (
            isinstance(array, torch.FloatTensor)
            or isinstance(array, torch.LongTensor)
            or isinstance(array, torch.ByteTensor)
        ):
            array = Variable(array, requires_grad=requires_grad)
        if "preserve_int" in kwargs and kwargs["preserve_int"] is True and is_int:
            array = array.long()
        array = set_cuda(array, is_cuda)
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def to_Variable_recur(item, type="float"):
    """Recursively transform numpy array into PyTorch tensor."""
    if isinstance(item, dict):
        return {key: to_Variable_recur(value, type=type) for key, value in item.items()}
    elif isinstance(item, tuple):
        return tuple(to_Variable_recur(element, type=type) for element in item)
    else:
        try:
            if type == "long":
                return torch.LongTensor(item)
            elif type == "float":
                return torch.FloatTensor(item)
            elif type == "bool":
                return torch.BoolTensor(item)
        except:
            return [to_Variable_recur(element, type=type) for element in item]


def to_Boolean(tensor):
    """Transform to Boolean tensor. For PyTorch version >= 1.2, use bool(). Otherwise use byte()"""
    version = torch.__version__
    version = eval(".".join(version.split(".")[:-1]))
    if version >= 1.2:
        return tensor.bool()
    else:
        return tensor.byte()


def init_module_weights(module_list, init_weights_mode="glorot-normal"):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        if init_weights_mode == "glorot-uniform":
            glorot_uniform_limit = np.sqrt(
                6 / float(module.in_features + module.out_features)
            )
            module.weight.data.uniform_(-glorot_uniform_limit, glorot_uniform_limit)
        elif init_weights_mode == "glorot-normal":
            glorot_normal_std = np.sqrt(
                2 / float(module.in_features + module.out_features)
            )
            module.weight.data.normal_(mean=0, std=glorot_normal_std)
        else:
            raise Exception(
                "init_weights_mode '{0}' not recognized!".format(init_weights_mode)
            )


def init_module_bias(module_list, init_bias_mode="zeros"):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        if init_bias_mode == "zeros":
            module.bias.data.fill_(0)
        else:
            raise Exception(
                "init_bias_mode '{0}' not recognized!".format(init_bias_mode)
            )


def init_weight(weight_list, init):
    """Initialize the weights"""
    if not isinstance(weight_list, list):
        weight_list = [weight_list]
    for weight in weight_list:
        if len(weight.size()) == 2:
            rows = weight.size(0)
            columns = weight.size(1)
        elif len(weight.size()) == 1:
            rows = 1
            columns = weight.size(0)
        if init is None:
            init = "glorot-normal"
        if not isinstance(init, str):
            weight.data.copy_(torch.FloatTensor(init))
        else:
            if init == "glorot-normal":
                glorot_normal_std = np.sqrt(2 / float(rows + columns))
                weight.data.normal_(mean=0, std=glorot_normal_std)
            else:
                raise Exception("init '{0}' not recognized!".format(init))


def init_bias(bias_list, init):
    """Initialize the bias"""
    if not isinstance(bias_list, list):
        bias_list = [bias_list]
    for bias in bias_list:
        if init is None:
            init = "zeros"
        if not isinstance(init, str):
            bias.data.copy_(torch.FloatTensor(init))
        else:
            if init == "zeros":
                bias.data.fill_(0)
            else:
                raise Exception("init '{0}' not recognized!".format(init))


def get_activation(activation):
    """Get activation"""
    assert activation in ACTIVATION_LIST + ["linear"]
    if activation == "linear":
        f = lambda x: x
    elif activation == "relu":
        f = F.relu
    elif activation == "leakyRelu":
        f = nn.LeakyReLU(negative_slope=0.3)
    elif activation == "leakyReluFlat":
        f = nn.LeakyReLU(negative_slope=0.01)
    elif activation == "tanh":
        f = torch.tanh
    elif activation == "softplus":
        f = F.softplus
    elif activation == "sigmoid":
        f = torch.sigmoid
    elif activation == "selu":
        f = F.selu
    elif activation == "elu":
        f = F.elu
    elif activation == "sign":
        f = lambda x: torch.sign(x)
    elif activation == "heaviside":
        f = lambda x: (torch.sign(x) + 1) / 2.0
    elif activation == "softmax":
        f = lambda x: nn.Softmax(dim=-1)(x)
    elif activation == "negLogSoftmax":
        f = lambda x: -torch.log(nn.Softmax(dim=-1)(x))
    elif activation == "naturalLogSoftmax":

        def natlogsoftmax(x):
            x = torch.cat((x, torch.zeros_like(x[..., :1])), axis=-1)
            norm = torch.logsumexp(x, axis=-1, keepdim=True)
            return -(x - norm)

        f = natlogsoftmax
    else:
        raise Exception("activation {0} not recognized!".format(activation))
    return f


def get_activation_noise(act_noise):
    noise_type = act_noise["type"]
    if noise_type == "identity":
        f = lambda x: x
    elif noise_type == "gaussian":
        f = lambda x: x + torch.randn(x.shape) * act_noise["scale"]
    elif noise_type == "uniform":
        f = lambda x: x + (torch.rand(x.shape) - 0.5) * act_noise["scale"]
    else:
        raise Exception("act_noise {} is not valid!".format(noise_type))
    return f


class MAELoss(_Loss):
    """Mean absolute loss"""

    def __init__(self, size_average=None, reduce=None):
        super(MAELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        assert not target.requires_grad, (
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )
        loss = (input - target).abs()
        if self.reduce:
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()
        return loss


class MultihotBinaryCrossEntropy(_Loss):
    """Multihot cross-entropy loss."""

    def __init__(self, size_average=None, reduce=None):
        super(MultihotBinaryCrossEntropy, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        assert not target.requires_grad, (
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )
        target = torch.tensor(
            np.concatenate(
                ([np.eye(10)[target[:, c]] for c in range(target.shape[1])]), axis=1
            )
        )
        return F.binary_cross_entropy_with_logits(input, target, reduce=self.reduce)


def get_criterion(loss_type, reduce=None, **kwargs):
    """Get loss function"""
    if loss_type == "huber":
        criterion = nn.SmoothL1Loss(reduce=reduce)
    elif loss_type == "mse":
        criterion = nn.MSELoss(reduce=reduce)
    elif loss_type == "mae":
        criterion = MAELoss(reduce=reduce)
    elif loss_type == "DL":
        criterion = Loss_Fun(
            core="DL",
            loss_precision_floor=kwargs["loss_precision_floor"]
            if "loss_precision_floor" in kwargs
            and kwargs["loss_precision_floor"] is not None
            else PrecisionFloorLoss,
            DL_sum=kwargs["DL_sum"] if "DL_sum" in kwargs else False,
        )
    elif loss_type == "DLs":
        criterion = Loss_Fun(
            core="DLs",
            loss_precision_floor=kwargs["loss_precision_floor"]
            if "loss_precision_floor" in kwargs
            and kwargs["loss_precision_floor"] is not None
            else PrecisionFloorLoss,
            DL_sum=kwargs["DL_sum"] if "DL_sum" in kwargs else False,
        )
    elif loss_type == "mlse":
        epsilon = 1e-10
        criterion = lambda pred, target: torch.log(
            nn.MSELoss(reduce=reduce)(pred, target) + epsilon
        )
    elif loss_type == "mse+mlse":
        epsilon = 1e-10
        criterion = (
            lambda pred, target: torch.log(
                nn.MSELoss(reduce=reduce)(pred, target) + epsilon
            )
            + nn.MSELoss(reduce=reduce)(pred, target).mean()
        )
    elif loss_type == "cross-entropy":
        criterion = nn.CrossEntropyLoss(reduce=reduce)
    elif loss_type == "multihot-bce":
        criterion = MultihotBinaryCrossEntropy(reduce=reduce)
    elif loss_type == "Loss_with_uncertainty":
        criterion = Loss_with_uncertainty(
            core=kwargs["loss_core"] if "loss_core" in kwargs else "mse", epsilon=1e-6
        )
    elif loss_type[:11] == "Contrastive":
        criterion_name = loss_type.split("-")[1]
        beta = eval(loss_type.split("-")[2])
        criterion = ContrastiveLoss(
            get_criterion(criterion_name, reduce=reduce, **kwargs), beta=beta
        )
    else:
        raise Exception("loss_type {0} not recognized!".format(loss_type))
    return criterion


def get_criteria_value(model, X, y, criteria_type, criterion, **kwargs):
    loss_precision_floor = (
        kwargs["loss_precision_floor"]
        if "loss_precision_floor" in kwargs
        else PrecisionFloorLoss
    )
    pred = forward(model, X, **kwargs)

    # Get loss:
    loss = to_np_array(criterion(pred, y))
    result = {"loss": loss}

    # Get DL:
    DL_type = criteria_type if "DL" in criteria_type else "DLs"
    data_DL = get_criterion(
        loss_type=DL_type, loss_precision_floor=loss_precision_floor, DL_sum=True
    )(pred, y)
    data_DL = to_np_array(data_DL)
    if not isinstance(model, list):
        model = [model]
    model_DL = np.sum([model_ele.DL for model_ele in model])
    DL = data_DL + model_DL
    result["DL"] = DL
    result["model_DL"] = model_DL
    result["data_DL"] = data_DL

    # Specify criteria_value:
    if criteria_type == "loss":
        criteria_value = loss
    elif "DL" in criteria_type:
        criteria_value = DL
    else:
        raise Exception("criteria type {0} not valid".format(criteria_type))
    print(result)
    return criteria_value, result


def get_optimizer(optim_type, lr, parameters, **kwargs):
    """Get optimizer"""
    momentum = kwargs["momentum"] if "momentum" in kwargs else 0
    if optim_type == "adam":
        amsgrad = kwargs["amsgrad"] if "amsgrad" in kwargs else False
        optimizer = optim.Adam(parameters, lr=lr, amsgrad=amsgrad)
    elif optim_type == "sgd":
        nesterov = kwargs["nesterov"] if "nesterov" in kwargs else False
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=nesterov)
    elif optim_type == "RMSprop":
        optimizer = optim.RMSprop(parameters, lr=lr, momentum=momentum)
    elif optim_type == "LBFGS":
        optimizer = optim.LBFGS(parameters, lr=lr)
    else:
        raise Exception("optim_type {0} not recognized!".format(optim_type))
    return optimizer


def get_full_struct_param_ele(struct_param, settings):
    struct_param_new = deepcopy(struct_param)
    for i, layer_struct_param in enumerate(struct_param_new):
        if settings is not None and layer_struct_param[1] != "Symbolic_Layer":
            layer_struct_param[2] = {
                key: value
                for key, value in deepcopy(settings).items()
                if key in ["activation"]
            }
            layer_struct_param[2].update(struct_param[i][2])
        else:
            layer_struct_param[2] = deepcopy(struct_param[i][2])
    return struct_param_new


def get_full_struct_param(struct_param, settings):
    struct_param_new_list = []
    if isinstance(struct_param, tuple):
        for i, struct_param_ele in enumerate(struct_param):
            if isinstance(settings, tuple):
                settings_ele = settings[i]
            else:
                settings_ele = settings
            struct_param_new_list.append(
                get_full_struct_param_ele(struct_param_ele, settings_ele)
            )
        return tuple(struct_param_new_list)
    else:
        return get_full_struct_param_ele(struct_param, settings)


class Early_Stopping(object):
    """Class for monitoring and suggesting early stopping"""

    def __init__(self, patience=100, epsilon=0, mode="min"):
        self.patience = patience
        self.epsilon = epsilon
        self.mode = mode
        self.best_value = None
        self.wait = 0

    def reset(self, value=None):
        self.best_value = value
        self.wait = 0

    def monitor(self, value):
        to_stop = False
        if self.patience is not None:
            if self.best_value is None:
                self.best_value = value
                self.wait = 0
            else:
                if (self.mode == "min" and value < self.best_value - self.epsilon) or (
                    self.mode == "max" and value > self.best_value + self.epsilon
                ):
                    self.best_value = value
                    self.wait = 0
                else:
                    if self.wait >= self.patience:
                        to_stop = True
                    else:
                        self.wait += 1
        return to_stop


class Performance_Monitor(object):
    def __init__(self, patience=100, epsilon=0, compare_mode="absolute"):
        self.patience = patience
        self.epsilon = epsilon
        self.compare_mode = compare_mode
        self.reset()

    def reset(self):
        self.best_value = None
        self.model_list = []
        self.wait = 0
        self.pivot_id = 0

    def monitor(self, value, **kwargs):
        to_stop = False
        is_accept = False
        if self.best_value is None:
            self.best_value = value
            self.wait = 0
            self.model_list = [deepcopy(kwargs)]
            log = deepcopy(self.model_list)
            is_accept = True
        else:
            self.model_list.append(deepcopy(kwargs))
            log = deepcopy(self.model_list)
            if self.compare_mode == "absolute":
                is_better = value <= self.best_value + self.epsilon
            elif self.compare_mode == "relative":
                is_better = value <= self.best_value * (1 + self.epsilon)
            else:
                raise
            if is_better:
                self.best_value = value
                self.pivot_id = self.pivot_id + 1 + self.wait
                self.wait = 0
                self.model_list = [deepcopy(kwargs)]
                is_accept = True
            else:
                if self.wait >= self.patience:
                    to_stop = True
                else:
                    self.wait += 1
        return (
            to_stop,
            deepcopy(self.model_list[0]),
            log,
            is_accept,
            deepcopy(self.pivot_id),
        )


def flatten(*tensors):
    """Flatten the tensor except the first dimension"""
    new_tensors = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            new_tensor = tensor.contiguous().view(tensor.shape[0], -1)
        elif isinstance(tensor, np.ndarray):
            new_tensor = tensor.reshape(tensor.shape[0], -1)
        else:
            print(new_tensor)
            raise Exception("tensors must be either torch.Tensor or np.ndarray!")
        new_tensors.append(new_tensor)
    if len(new_tensors) == 1:
        new_tensors = new_tensors[0]
    return new_tensors


def expand_indices(vector, expand_size):
    """Expand each element ele in the vector to range(ele * expand_size, (ele + 1) * expand_size)"""
    assert isinstance(vector, torch.Tensor)
    vector *= expand_size
    vector_expand = [vector + i for i in range(expand_size)]
    vector_expand = torch.stack(vector_expand, 0)
    vector_expand = vector_expand.transpose(0, 1).contiguous().view(-1)
    return vector_expand


def to_one_hot(idx, num):
    """Transform a 1D vector into a one-hot vector with num classes"""
    if len(idx.size()) == 1:
        idx = idx.unsqueeze(-1)
    if not isinstance(idx, Variable):
        if isinstance(idx, np.ndarray):
            idx = torch.LongTensor(idx)
        idx = Variable(idx, requires_grad=False)
    onehot = Variable(torch.zeros(idx.size(0), num), requires_grad=False)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx, 1)
    return onehot


def train_test_split(*args, test_size=0.1):
    """Split the dataset into training and testing sets"""
    import torch

    num_examples = len(args[0])
    train_list = []
    test_list = []
    if test_size is not None:
        num_test = int(num_examples * test_size)
        num_train = num_examples - num_test
        idx_train = np.random.choice(range(num_examples), size=num_train, replace=False)
        idx_test = set(range(num_examples)) - set(idx_train)
        device = args[0].device
        idx_train = torch.LongTensor(list(idx_train)).to(device)
        idx_test = torch.LongTensor(list(idx_test)).to(device)
        for arg in args:
            train_list.append(arg[idx_train])
            test_list.append(arg[idx_test])
    else:
        train_list = args
        test_list = args
    return train_list, test_list


def make_dir(filename):
    """Make directory using filename if the directory does not exist"""
    import os
    import errno

    if not os.path.exists(os.path.dirname(filename)):
        print(
            "directory {0} does not exist, created.".format(os.path.dirname(filename))
        )
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
            raise


def get_accuracy(pred, target):
    """Get accuracy from prediction and target"""
    assert len(pred.shape) == len(target.shape) == 1
    assert len(pred) == len(target)
    pred, target = to_np_array(pred, target)
    accuracy = (pred == target).sum().astype(float) / len(pred)
    return accuracy


def get_model_accuracy(model, X, y, **kwargs):
    """Get accuracy from model, X and target"""
    is_tensor = kwargs["is_tensor"] if "is_tensor" in kwargs else False
    pred = model(X)
    assert len(pred.shape) == 2
    assert isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor)
    assert len(y.shape) == 1
    pred_max = pred.max(-1)[1]
    acc = (y == pred_max).float().mean()
    if not is_tensor:
        acc = to_np_array(acc)
    return acc


def normalize_tensor(X, new_range=None, mean=None, std=None):
    """Normalize the tensor's value range to new_range"""
    X = X.float()
    if new_range is not None:
        assert mean is None and std is None
        X_min, X_max = X.min().item(), X.max().item()
        X_normalized = (X - X_min) / float(X_max - X_min)
        X_normalized = X_normalized * (new_range[1] - new_range[0]) + new_range[0]
    else:
        X_mean = X.mean().item()
        X_std = X.std().item()
        X_normalized = (X - X_mean) / X_std
        X_normalized = X_normalized * std + mean
    return X_normalized


def try_eval(string):
    """Try to evaluate a string. If failed, use original string."""
    try:
        return eval(string)
    except:
        return string


def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def get_args(arg, arg_id=1, type="str"):
    """get sys arguments from either command line or Jupyter"""
    try:
        get_ipython().run_line_magic("matplotlib", "inline")
        arg_return = arg
    except:
        import sys

        try:
            arg_return = sys.argv[arg_id]
            if type == "int":
                arg_return = int(arg_return)
            elif type == "float":
                arg_return = float(arg_return)
            elif type == "bool":
                arg_return = eval(arg_return)
            elif type == "eval":
                arg_return = eval(arg_return)
            elif type == "tuple":
                arg_return = eval_tuple(arg_return)
            elif type == "str":
                pass
            else:
                raise Exception("type {0} not recognized!".format(type))
        except:
            #             raise
            arg_return = arg
    return arg_return


def forward(model, X, **kwargs):
    autoencoder = kwargs["autoencoder"] if "autoencoder" in kwargs else None
    is_Lagrangian = kwargs["is_Lagrangian"] if "is_Lagrangian" in kwargs else False
    output = X
    if not is_Lagrangian:
        if isinstance(model, list) or isinstance(model, tuple):
            for model_ele in model:
                output = model_ele(output)
        else:
            output = model(output)
    else:
        if isinstance(model, list) or isinstance(model, tuple):
            for i, model_ele in enumerate(model):
                if i != len(model) - 1:
                    output = model_ele(output)
                else:
                    output = get_Lagrangian_loss(model_ele, output)
        else:
            output = get_Lagrangian_loss(model, output)
    if autoencoder is not None:
        output = autoencoder.decode(output)
    return output


def logplus(x):
    return torch.clamp(torch.log(torch.clamp(x, 1e-9)) / np.log(2), 0)


class Loss_Fun(nn.Module):
    def __init__(
        self,
        core="mse",
        epsilon=1e-10,
        loss_precision_floor=PrecisionFloorLoss,
        DL_sum=False,
    ):
        super(Loss_Fun, self).__init__()
        self.name = "Loss_Fun"
        self.core = core
        self.epsilon = epsilon
        self.loss_precision_floor = loss_precision_floor
        self.DL_sum = DL_sum

    def forward(self, pred, target, sample_weights=None, is_mean=True):
        if len(pred.size()) == 3:
            pred = pred.squeeze(1)
        assert pred.size() == target.size(), "pred and target must have the same size!"
        if self.core == "huber":
            loss = nn.SmoothL1Loss(reduce=False)(pred, target)
        elif self.core == "mse":
            loss = get_criterion(self.core, reduce=False)(pred, target)
        elif self.core == "mae":
            loss = get_criterion(self.core, reduce=False)(pred, target)
        elif self.core == "mse-conv":
            loss = (
                get_criterion(self.core, reduce=False)(pred, target).mean(-1).mean(-1)
            )
        elif self.core == "mlse":
            loss = torch.log(nn.MSELoss(reduce=False)(pred, target) + self.epsilon)
        elif self.core == "mse+mlse":
            loss = (
                torch.log(nn.MSELoss(reduce=False)(pred, target) + self.epsilon)
                + nn.MSELoss(reduce=False)(pred, target).mean()
            )
        elif self.core == "DL":
            loss = logplus(
                MAELoss(reduce=False)(pred, target) / self.loss_precision_floor
            )
        elif self.core == "DLs":
            loss = torch.log(
                1
                + nn.MSELoss(reduce=False)(pred, target)
                / self.loss_precision_floor ** 2
            ) / np.log(4)
        else:
            raise Exception("loss mode {0} not recognized!".format(self.core))
        if (
            len(loss.shape) == 4
        ):  # Dealing with pixel inputs of (num_examples, channels, height, width)
            loss = loss.mean(-1).mean(-1)
        if loss.size(-1) > 1:
            loss = loss.sum(-1, keepdim=True)
        if sample_weights is not None:
            assert tuple(loss.size()) == tuple(sample_weights.size())
            loss = loss * sample_weights
        if is_mean:
            if sample_weights is not None:
                loss = loss * len(sample_weights) / sample_weights.sum()
            if self.DL_sum:
                loss = loss.sum()
            else:
                loss = loss.mean()
        return loss


class Loss_with_uncertainty(nn.Module):
    def __init__(self, core="mse", epsilon=1e-6):
        super(Loss_with_uncertainty, self).__init__()
        self.name = "Loss_with_uncertainty"
        self.core = core
        self.epsilon = epsilon

    def forward(
        self, pred, target, log_std=None, std=None, sample_weights=None, is_mean=True
    ):
        if self.core == "mse":
            loss_core = get_criterion(self.core, reduce=False)(pred, target) / 2
        elif self.core == "mae":
            loss_core = get_criterion(self.core, reduce=False)(pred, target)
        elif self.core == "huber":
            loss_core = get_criterion(self.core, reduce=False)(pred, target)
        elif self.core == "mlse":
            loss_core = torch.log((target - pred) ** 2 + 1e-10)
        elif self.core == "mse+mlse":
            loss_core = (target - pred) ** 2 / 2 + torch.log(
                (target - pred) ** 2 + 1e-10
            )
        else:
            raise Exception("loss's core {0} not recognized!".format(self.core))
        if std is not None:
            assert log_std is None
            loss = loss_core / (self.epsilon + std ** 2) + torch.log(std + 1e-7)
        else:
            loss = loss_core / (self.epsilon + torch.exp(2 * log_std)) + log_std
        if sample_weights is not None:
            sample_weights = sample_weights.view(loss.size())
            loss = loss * sample_weights
        if is_mean:
            loss = loss.mean()
        return loss


def expand_tensor(tensor, dim, times):
    """Repeat the value of a tensor locally along the given dimension"""
    if isinstance(times, int) and times == 1:
        return tensor
    if dim < 0:
        dim += len(tensor.size())
    assert dim >= 0
    size = list(tensor.size())
    repeat_times = [1] * (len(size) + 1)
    repeat_times[dim + 1] = times
    size[dim] = size[dim] * times
    return tensor.unsqueeze(dim + 1).repeat(repeat_times).view(*size)


def sample(dist, n=None):
    """Sample n instances from distribution dist"""
    if n is None:
        return dist.rsample()
    else:
        return dist.rsample((n,))


def set_variational_output_size(model_dict, reparam_mode, latent_size):
    if reparam_mode.startswith("full"):
        model_dict["struct_param"][-1][0] = int((latent_size + 3) * latent_size / 2)
    elif reparam_mode.startswith("diag"):
        if not reparam_mode.startswith("diagg"):
            if model_dict["type"] == "Mixture_Model":
                for i in range(model_dict["num_components"]):
                    if isinstance(model_dict["model_dict_list"], list):
                        model_dict["model_dict_list"][i]["struct_param"][-1][0] = (
                            2 * latent_size
                        )
                    elif isinstance(model_dict["model_dict_list"], dict):
                        model_dict["model_dict_list"]["struct_param"][-1][0] = (
                            2 * latent_size
                        )
                    else:
                        raise
            else:
                model_dict["struct_param"][-1][0] = 2 * latent_size
    else:
        raise


def shrink_tensor(tensor, dim, shrink_ratio, mode="any"):
    """Shrink a tensor along certain dimension using neighboring sites"""
    is_tensor = isinstance(tensor, torch.Tensor)
    shape = tuple(tensor.shape)
    if dim < 0:
        dim += len(tensor.shape)
    assert shape[dim] % shrink_ratio == 0
    new_dim = int(shape[dim] / shrink_ratio)
    new_shape = shape[:dim] + (new_dim, shrink_ratio) + shape[dim + 1 :]
    if is_tensor:
        new_tensor = tensor.view(*new_shape)
    else:
        new_tensor = np.reshape(tensor, new_shape)
    if mode == "any":
        assert tensor.dtype == "bool" or isinstance(tensor, torch.ByteTensor)
        return new_tensor.any(dim + 1)
    elif mode == "all":
        assert tensor.dtype == "bool" or isinstance(tensor, torch.ByteTensor)
        return new_tensor.all(dim + 1)
    elif mode == "sum":
        return new_tensor.sum(dim + 1)
    elif mode == "mean":
        return new_tensor.mean(dim + 1)
    else:
        raise


def permute_dim(X, dim, idx, group_sizes, mode="permute"):
    from copy import deepcopy

    assert dim != 0
    device = X.device
    if isinstance(idx, tuple) or isinstance(idx, list):
        k, ll = idx
        X_permute = X[:, k, ll * group_sizes : (ll + 1) * group_sizes]
        num = X_permute.size(0)
        if mode == "permute":
            new_idx = torch.randperm(num).to(device)
        elif mode == "resample":
            new_idx = torch.randint(num, size=(num,)).long().to(device)
        else:
            raise
        X_permute = X_permute.index_select(0, new_idx)
        X_new = deepcopy(X)
        X_new[:, k, ll * group_sizes : (ll + 1) * group_sizes] = X_permute
    else:
        X_permute = X.index_select(
            dim,
            torch.arange(idx * group_sizes, (idx + 1) * group_sizes).long().to(device),
        )
        num = X_permute.size(0)
        if mode == "permute":
            new_idx = torch.randperm(num).to(device)
        elif mode == "resample":
            new_idx = torch.randint(num, size=(num,)).long().to(device)
        else:
            raise
        X_permute = X_permute.index_select(0, new_idx)
        X_new = deepcopy(X)
        if dim == 1:
            X_new[:, idx * group_sizes : (idx + 1) * group_sizes] = X_permute
        elif dim == 2:
            X_new[:, :, idx * group_sizes : (idx + 1) * group_sizes] = X_permute
        elif dim == 3:
            X_new[:, :, :, idx * group_sizes : (idx + 1) * group_sizes] = X_permute
        else:
            raise
    return X_new


def fill_triangular(vec, dim, mode="lower"):
    """Fill an lower or upper triangular matrices with given vectors"""
    num_examples, size = vec.shape
    assert size == dim * (dim + 1) // 2
    matrix = torch.zeros(num_examples, dim, dim).to(vec.device)
    idx = (torch.tril(torch.ones(dim, dim)) == 1).unsqueeze(0)
    idx = idx.repeat(num_examples, 1, 1)
    if mode == "lower":
        matrix[idx] = vec.contiguous().view(-1)
    elif mode == "upper":
        matrix[idx] = vec.contiguous().view(-1)
    else:
        raise Exception("mode {0} not recognized!".format(mode))
    return matrix


def get_loss_cumu(loss_dict, cumu_mode):
    """Combine different losses to obtain a single scalar loss"""
    if cumu_mode == "original":
        return loss_dict
    if isinstance(loss_dict, dict):
        loss_list = torch.stack([loss for loss in loss_dict.values()])
    elif isinstance(loss_dict, list):
        loss_list = torch.stack(loss_dict)
    else:
        raise
    N = len(loss_list)
    epsilon = 1e-20  # to prevent NaN
    if cumu_mode[0] == "generalized-mean":
        if cumu_mode[1] == -1:
            cumu_mode = "harmonic"
        elif cumu_mode[1] == 0:
            cumu_mode = "geometric"
        elif cumu_mode[1] == 1:
            cumu_mode = "mean"

    if cumu_mode == "harmonic":
        loss = N / (1 / (loss_list + epsilon)).sum()
    elif cumu_mode == "geometric":
        loss = (loss_list + epsilon).prod() ** (1 / float(N))
    elif cumu_mode == "mean":
        loss = loss_list.mean()
    elif cumu_mode == "sum":
        loss = loss_list.sum()
    elif cumu_mode == "min":
        loss = loss_list.min()
    elif cumu_mode[0] == "generalized-mean":
        order = cumu_mode[1]
        loss = (((loss_list + epsilon) ** order).mean()) ** (1 / float(order))
    else:
        raise
    return loss


def matrix_diag_transform(matrix, fun):
    """Return the matrices whose diagonal elements have been executed by the function 'fun'."""
    num_examples = len(matrix)
    idx = torch.eye(matrix.size(-1)).byte().unsqueeze(0)
    idx = idx.repeat(num_examples, 1, 1)
    new_matrix = matrix.clone()
    new_matrix[idx] = fun(matrix.diagonal(dim1=1, dim2=2).contiguous().view(-1))
    return new_matrix


def sort_two_lists(list1, list2, reverse=False):
    """Sort two lists according to the first list."""
    if reverse:
        List = deepcopy(
            [
                list(x)
                for x in zip(
                    *sorted(
                        zip(deepcopy(list1), deepcopy(list2)),
                        key=operator.itemgetter(0),
                        reverse=True,
                    )
                )
            ]
        )
    else:
        List = deepcopy(
            [
                list(x)
                for x in zip(
                    *sorted(
                        zip(deepcopy(list1), deepcopy(list2)),
                        key=operator.itemgetter(0),
                    )
                )
            ]
        )
    if len(List) == 0:
        return [], []
    else:
        return List[0], List[1]


def sort_dict(Dict, reverse=False):
    """Return an ordered dictionary whose values are sorted"""
    from collections import OrderedDict

    orderedDict = OrderedDict()
    keys, values = list(Dict.keys()), list(Dict.values())
    values_sorted, keys_sorted = sort_two_lists(values, keys, reverse=reverse)
    for key, value in zip(keys_sorted, values_sorted):
        orderedDict[key] = value
    return orderedDict


def get_dict_items(Dict, idx):
    """Obtain dictionary items with the current ordering of dictionary keys"""
    from collections import OrderedDict
    from copy import deepcopy

    keys = list(Dict.keys())
    new_dict = OrderedDict()
    for id in idx:
        new_dict[keys[id]] = deepcopy(Dict[keys[id]])
    return new_dict


def to_string(List, connect="-", num_digits=None, num_strings=None):
    """Turn a list into a string, with specified format"""
    if not isinstance(List, list) and not isinstance(List, tuple):
        return List
    if num_strings is None:
        if num_digits is None:
            return connect.join([str(element) for element in List])
        else:
            return connect.join(
                ["{0:.{1}f}".format(element, num_digits) for element in List]
            )
    else:
        if num_digits is None:
            return connect.join([str(element)[:num_strings] for element in List])
        else:
            return connect.join(
                [
                    "{0:.{1}f}".format(element, num_digits)[:num_strings]
                    for element in List
                ]
            )


def split_string(string):
    """Given a string, return the core string and the number suffix.
    If there is no number suffix, the num_core will be None.
    """
    # Get the starting index for the number suffix:
    assert isinstance(string, str)
    i = 1
    for i in range(1, len(string) + 1):
        if string[-i] in [str(k) for k in range(10)]:
            continue
        else:
            break
    idx = len(string) - i + 1
    # Obtain string_core and num_core:
    string_core = string[:idx]
    if len(string[idx:]) > 0:
        num_core = eval(string[idx:])
    else:
        num_core = None

    return string_core, num_core


def canonicalize_strings(operators):
    """Given a list of strings, return the canonical version.
    
    Example:
        operators = ["EqualRow1", "EqualRow2", "EqualWidth3"]
    
        Returns: mapping = {'EqualRow1': 'EqualRow',
                            'EqualRow2': 'EqualRow1',
                            'EqualWidth3': 'EqualWidth'}
    """
    operators_core = [split_string(ele)[0] for ele in operators]
    counts = Counter(operators_core)
    new_counts = {key: 0 for key in counts}
    mapping = {}
    for operator, operator_core in zip(operators, operators_core):
        count = new_counts[operator_core]
        if count == 0:
            mapping[operator] = operator_core
        else:
            mapping[operator] = "{}{}".format(operator_core, count)
        new_counts[operator_core] += 1
    return mapping


def get_rename_mapping(base_keys, adding_keys):
    """Given a list of base_keys and adding keys, return a mapping of how to 
    rename adding_keys s.t. adding_keys do not have name conflict with base_keys.
    """
    mapping = {}
    for key in adding_keys:
        if key in base_keys:
            string_core, num_core = split_string(key)
            num_proposed = num_core + 1 if num_core is not None else 1
            proposed_name = "{}{}".format(string_core, num_proposed)
            while proposed_name in base_keys:
                num_proposed += 1
                proposed_name = "{}{}".format(string_core, num_proposed)
            mapping[key] = proposed_name
    return mapping


def view_item(dict_list, key):
    if not isinstance(key, tuple):
        return [element[key] for element in dict_list]
    else:
        return [element[key[0]][key[1]] for element in dict_list]


def filter_filename(dirname, include=[], exclude=[], array_id=None):
    """Filter filename in a directory"""

    def get_array_id(filename):
        array_id = filename.split("_")[-2]
        try:
            array_id = eval(array_id)
        except:
            pass
        return array_id

    filename_collect = []
    if array_id is None:
        filename_cand = [filename for filename in os.listdir(dirname)]
    else:
        filename_cand = [
            filename
            for filename in os.listdir(dirname)
            if get_array_id(filename) == array_id
        ]

    if not isinstance(include, list):
        include = [include]
    if not isinstance(exclude, list):
        exclude = [exclude]

    for filename in filename_cand:
        is_in = True
        for element in include:
            if element not in filename:
                is_in = False
                break
        for element in exclude:
            if element in filename:
                is_in = False
                break
        if is_in:
            filename_collect.append(filename)
    return filename_collect


def sort_filename(filename_list):
    """Sort the files according to the id at the end. The filename is in the form of *_NUMBER.p """
    iter_list = []
    for filename in filename_list:
        iter_num = eval(filename.split("_")[-1].split(".")[0])
        iter_list.append(iter_num)
    iter_list_sorted, filename_list_sorted = sort_two_lists(
        iter_list, filename_list, reverse=True
    )
    return filename_list_sorted


def remove_files_in_directory(directory, is_remove_subdir=False):
    """Remove files in a directory"""
    import os, shutil

    if directory is None:
        return
    if not os.path.isdir(directory):
        print("Directory {0} does not exist!".format(directory))
        return
    for the_file in os.listdir(directory):
        file_path = os.path.join(directory, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif is_remove_subdir and os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def softmax(tensor, dim):
    assert isinstance(tensor, np.ndarray)
    tensor = tensor - tensor.mean(dim, keepdims=True)
    tensor = np.exp(tensor)
    tensor = tensor / tensor.sum(dim, keepdims=True)
    return tensor


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
          From https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
    """

    @staticmethod
    def forward(ctx, input):
        m = input.detach().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).type_as(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.numpy().astype(np.float_)
            gm = grad_output.data.numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def get_flat_function(List, idx):
    """Get the idx index of List. If idx >= len(List), return the last element"""
    if idx < 0:
        return List[0]
    elif idx < len(List):
        return List[idx]
    else:
        return List[-1]


def Beta_Function(x, alpha, beta):
    """Beta function"""
    from scipy.special import gamma

    return (
        gamma(alpha + beta)
        / gamma(alpha)
        / gamma(beta)
        * x ** (alpha - 1)
        * (1 - x) ** (beta - 1)
    )


def Zip(*data, **kwargs):
    """Recursive unzipping of data structure
    Example: Zip(*[(('a',2), 1), (('b',3), 2), (('c',3), 3), (('d',2), 4)])
    ==> [[['a', 'b', 'c', 'd'], [2, 3, 3, 2]], [1, 2, 3, 4]]
    Each subtree in the original data must be in the form of a tuple.
    In the **kwargs, you can set the function that is applied to each fully unzipped subtree.
    """
    import collections

    function = kwargs["function"] if "function" in kwargs else None
    if len(data) == 1:
        return data[0]
    data = [list(element) for element in zip(*data)]
    for i, element in enumerate(data):
        if isinstance(element[0], tuple):
            data[i] = Zip(*element, **kwargs)
        elif isinstance(element, list):
            if function is not None:
                data[i] = function(element)
    return data


class Gradient_Noise_Scale_Gen(object):
    def __init__(
        self,
        epochs,
        gamma=0.55,
        eta=0.01,
        noise_scale_start=1e-2,
        noise_scale_end=1e-6,
        gradient_noise_interval_epoch=1,
        fun_pointer="generate_scale_simple",
    ):
        self.epochs = epochs
        self.gradient_noise_interval_epoch = gradient_noise_interval_epoch
        self.max_iter = int(self.epochs / self.gradient_noise_interval_epoch) + 1
        self.gamma = gamma
        self.eta = eta
        self.noise_scale_start = noise_scale_start
        self.noise_scale_end = noise_scale_end
        self.generate_scale = getattr(
            self, fun_pointer
        )  # Sets the default function to generate scale

    def generate_scale_simple(self, verbose=True):
        gradient_noise_scale = np.sqrt(
            self.eta * (np.array(range(self.max_iter)) + 1) ** (-self.gamma)
        )
        if verbose:
            print(
                "gradient_noise_scale: start = {0}, end = {1:.6f}, gamma = {2}, length = {3}".format(
                    gradient_noise_scale[0],
                    gradient_noise_scale[-1],
                    self.gamma,
                    self.max_iter,
                )
            )
        return gradient_noise_scale

    def generate_scale_fix_ends(self, verbose=True):
        ratio = (self.noise_scale_start / float(self.noise_scale_end)) ** (
            1 / self.gamma
        ) - 1
        self.bb = self.max_iter / ratio
        self.aa = self.noise_scale_start * self.bb ** self.gamma
        gradient_noise_scale = np.sqrt(
            self.aa * (np.array(range(self.max_iter)) + self.bb) ** (-self.gamma)
        )
        if verbose:
            print(
                "gradient_noise_scale: start = {0}, end = {1:.6f}, gamma = {2}, length = {3}".format(
                    gradient_noise_scale[0],
                    gradient_noise_scale[-1],
                    self.gamma,
                    self.max_iter,
                )
            )
        return gradient_noise_scale


def load_model(filename, mode="pickle"):
    if mode == "pickle":
        model_dict = pickle.load(open(filename, "rb"))
    elif mode == "json":
        with open(filename, "r") as outfile:
            json_dict = json.load(outfile)
            model_dict = deserialize(json_dict)
    else:
        raise Exception("mode {} is not valid!".format(mode))
    return model_dict


def save_model(model_dict, filename, mode="pickle"):
    if mode == "pickle":
        pickle.dump(model_dict, open(filename, "wb"))
    elif mode == "json":
        with open(filename, "w") as outfile:
            json.dump(serialize(model_dict), outfile)
    else:
        raise Exception("mode {} is not valid!".format(mode))


def to_cpu_recur(item, to_target=None):
    if isinstance(item, dict):
        return {
            key: to_cpu_recur(value, to_target=to_target) for key, value in item.items()
        }
    elif isinstance(item, list):
        return [to_cpu_recur(element, to_target=to_target) for element in item]
    elif isinstance(item, tuple):
        return tuple(to_cpu_recur(element, to_target=to_target) for element in item)
    elif isinstance(item, set):
        return {to_cpu_recur(element, to_target=to_target) for element in item}
    else:
        if isinstance(item, torch.Tensor):
            if item.is_cuda:
                item = item.cpu()
            if to_target is not None and to_target == "np":
                item = item.detach().numpy()
            return item
        if to_target is not None and to_target == "torch":
            if isinstance(item, np.ndarray):
                item = torch.FloatTensor(item)
                return item
        return item


def serialize(item):
    if isinstance(item, dict):
        return {str(key): serialize(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [serialize(element) for element in item]
    elif isinstance(item, tuple):
        return tuple(serialize(element) for element in item)
    elif isinstance(item, np.ndarray):
        return item.tolist()
    else:
        return str(item)


def deserialize(item):
    if isinstance(item, dict):
        try:
            return {eval(key): deserialize(value) for key, value in item.items()}
        except:
            return {key: deserialize(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [deserialize(element) for element in item]
    elif isinstance(item, tuple):
        return tuple(deserialize(element) for element in item)
    else:
        if isinstance(item, str) and item in CLASS_TYPES:
            return item
        else:
            try:
                return eval(item)
            except:
                return item


def deserialize_list_str(string):
    """Parse a string version of recursive lists into recursive lists of strings."""
    assert string.startswith("[") and string.endswith("]")
    string = string[1:-1]
    bracket_count = 0
    punctuations = [0]
    for i, letter in enumerate(string):
        if letter == "," and bracket_count == 0:
            punctuations.append(i)
        elif letter == "[":
            bracket_count += 1
        elif letter == "]":
            bracket_count -= 1

    if len(punctuations) == 1:
        return [string]
    List = []
    for i in range(len(punctuations)):
        if i == 0:
            element = string[: punctuations[1]]
        elif i == len(punctuations) - 1:
            element = string[punctuations[i] + 1 :].strip()
        else:
            element = string[punctuations[i] + 1 : punctuations[i + 1]].strip()
        if element.startswith("[") and element.endswith("]"):
            List.append(deserialize_list_str(element))
        else:
            List.append(element)
    return List


def flatten_list(graph):
    """Flatten a recursive list of lists into a flattened list."""
    if not isinstance(graph, list):
        return [deepcopy(graph)]
    else:
        flattened_graph = []
        for subgraph in graph:
            flattened_graph += flatten_list(subgraph)
        return flattened_graph


def get_num_params(model, is_trainable=None):
    """Get number of parameters of the model, specified by 'None': all parameters;
    True: trainable parameters; False: non-trainable parameters.
    """
    num_params = 0
    for param in list(model.parameters()):
        nn = 1
        if (
            is_trainable is None
            or (is_trainable is True and param.requires_grad is True)
            or (is_trainable is False and param.requires_grad is False)
        ):
            for s in list(param.size()):
                nn = nn * s
            num_params += nn
    return num_params


def set_subtract(list1, list2):
    list1 = list(list1)
    list2 = list(list2)
    assert isinstance(list1, list)
    assert isinstance(list2, list)
    return list(set(list1) - set(list2))


def find_nearest(array, value, mode="abs"):
    array = deepcopy(array)
    array = np.asarray(array)
    if mode == "abs":
        idx = (np.abs(array - value)).argmin()
    elif mode == "le":
        array[array > value] = -np.Inf
        idx = array.argmax()
    elif mode == "ge":
        array[array < value] = np.Inf
        idx = array.argmin()
    else:
        raise
    return idx, array[idx]


def sort_matrix(matrix, dim, reverse=False):
    if dim == 0:
        _, idx_sort = sort_two_lists(
            matrix[:, 0], range(len(matrix[:, 0])), reverse=reverse
        )
        return matrix[idx_sort]
    elif dim == 1:
        _, idx_sort = sort_two_lists(
            matrix[0, :], range(len(matrix[0, :])), reverse=reverse
        )
        return matrix[:, idx_sort]
    else:
        raise


def hashing(X, width=128):
    import hashlib

    def hash_ele(x):
        return np.array(
            [
                int(element)
                for element in np.binary_repr(
                    int(hashlib.md5(x.view(np.uint8)).hexdigest(), 16), width=128
                )
            ]
        )[-width:]

    is_torch = isinstance(X, torch.Tensor)
    if is_torch:
        is_cuda = X.is_cuda
    X = to_np_array(X)
    hash_list = np.array([hash_ele(x) for x in X])
    if is_torch:
        hash_list = to_Variable(hash_list, is_cuda=is_cuda)

    # Check collision:
    string = ["".join([str(e) for e in ele]) for ele in to_np_array(hash_list)]
    uniques = np.unique(np.unique(string, return_counts=True)[1], return_counts=True)
    return hash_list, uniques


def pplot(
    X,
    y,
    markers=".",
    label=None,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=(10, 8),
    fontsize=18,
    plt=None,
    is_show=True,
):
    if plt is None:
        import matplotlib.pylab as plt

        plt.figure(figsize=figsize)
    plt.plot(to_np_array(X), to_np_array(y), markers, label=label)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    if label is not None:
        plt.legend(fontsize=fontsize)
    if is_show:
        plt.show()
    return plt


def formalize_value(value, precision):
    """Formalize value with floating or scientific notation, depending on its absolute value."""
    if 10 ** (-(precision - 1)) <= np.abs(value) <= 10 ** (precision - 1):
        return "{0:.{1}f}".format(value, precision)
    else:
        return "{0:.{1}e}".format(value, precision)


def plot1D_3(
    X_mesh, Z_mesh, target, view_init=[(30, 50), (90, -90), (0, 0)], zlabel=None
):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm

    X_mesh, Z_mesh, target = to_np_array(X_mesh, Z_mesh, target)
    fig = plt.figure(figsize=(22, 7))
    ax = fig.add_subplot(131, projection="3d")
    ax.plot_surface(
        X_mesh,
        Z_mesh,
        target,
        alpha=0.5,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Z Label")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=view_init[0][0], azim=view_init[0][1])

    ax = fig.add_subplot(132, projection="3d")
    ax.plot_surface(
        X_mesh,
        Z_mesh,
        target,
        alpha=0.5,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Z Label")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=view_init[1][0], azim=view_init[1][1])

    ax = fig.add_subplot(133, projection="3d")
    ax.plot_surface(
        X_mesh,
        Z_mesh,
        target,
        alpha=0.5,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Z Label")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=view_init[2][0], azim=view_init[2][1])
    plt.show()


class RampupLR(_LRScheduler):
    """Ramp up the learning rate in exponential steps."""

    def __init__(self, optimizer, num_steps=200, last_epoch=-1):
        self.num_steps = num_steps
        super(RampupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * np.logspace(-12, 0, self.num_steps + 1)[self.last_epoch]
            for base_lr in self.base_lrs
        ]


def set_cuda(tensor, is_cuda):
    if isinstance(is_cuda, str):
        return tensor.cuda(is_cuda)
    else:
        if is_cuda:
            return tensor.cuda()
        else:
            return tensor


def isin(ar1, ar2):
    ar2 = torch.LongTensor(ar2)
    ar2 = ar2.to(ar1.device)
    return (ar1[..., None] == ar2).any(-1)


def filter_labels(X, y, labels):
    idx = isin(y, labels)
    return X[idx], y[idx]


def argmin_random(tensor):
    """Returns the flattened argmin of the tensor, and tie-breaks using random choice."""
    argmins = np.flatnonzero(tensor == tensor.min())
    if len(argmins) > 0:
        return np.random.choice(argmins)
    else:
        return np.NaN


def argmax_random(tensor):
    """Returns the flattened argmax of the tensor, and tie-breaks using random choice."""
    argmaxs = np.flatnonzero(tensor == tensor.max())
    if len(argmins) > 0:
        return np.random.choice(argmaxs)
    else:
        return np.NaN


class Transform_Label(object):
    def __init__(self, label_noise_matrix=None, is_cuda=False):
        self.label_noise_matrix = label_noise_matrix
        if self.label_noise_matrix is not None:
            assert ((self.label_noise_matrix.sum(0) - 1) < 1e-10).all()
        self.device = torch.device(
            is_cuda if isinstance(is_cuda, str) else "cuda" if is_cuda else "cpu"
        )

    def __call__(self, y):
        if self.label_noise_matrix is None:
            return y
        else:
            noise_matrix = self.label_noise_matrix
            dim = len(noise_matrix)
            y_tilde = []
            for y_ele in y:
                flip_rate = noise_matrix[:, y_ele]
                y_ele = np.random.choice(dim, p=flip_rate)
                y_tilde.append(y_ele)
            y_tilde = torch.LongTensor(y_tilde).to(self.device)
            return y_tilde


def base_repr(n, base, length):
    assert n < base ** length, "n should be smaller than b ** length"
    base_repr_str = np.base_repr(n, base, padding=length)[-length:]
    return [int(ele) for ele in base_repr_str]


def base_repr_2_int(List, base):
    if len(List) == 1:
        return List[0]
    elif len(List) == 0:
        return 0
    else:
        return base * base_repr_2_int(List[:-1], base) + List[-1]


def get_variable_name_list(expressions):
    """Get variable names from a given expressions list"""
    return sorted(
        list(
            {
                symbol.name
                for expression in expressions
                for symbol in expression.free_symbols
                if "x" in symbol.name
            }
        )
    )


def get_param_name_list(expressions):
    """Get parameter names from a given expressions list"""
    return sorted(
        list(
            {
                symbol.name
                for expression in expressions
                for symbol in expression.free_symbols
                if "x" not in symbol.name
            }
        )
    )


def get_function_name_list(symbolic_expression):
    from sympy import Function
    from sympy.utilities.lambdify import implemented_function

    symbolic_expression = standardize_symbolic_expression(symbolic_expression)
    function_name_list = list(
        {
            element.func.__name__
            for expression in symbolic_expression
            for element in expression.atoms(Function)
            if element.func.__name__ not in ["linear"]
        }
    )
    implemented_function = {}
    for function_name in function_name_list:
        try:
            implemented_function[function_name] = implemented_function(
                Function(function_name), get_activation(function_name)
            )
        except:
            pass
    return function_name_list, implemented_function


def substitute(expressions, param_dict):
    """Substitute each expression in the expression using the param_dict"""
    new_expressions = []
    has_param_list = []
    for expression in expressions:
        has_param = len(get_param_name_list([expression])) > 0
        has_param_list.append(has_param)
        new_expressions.append(expression.subs(param_dict))
    return new_expressions, has_param_list


def get_coeffs(expression):
    """Get coefficients as a list from an expression, w.r.t. its sorted variable name list"""
    from sympy import Poly

    variable_names = get_variable_name_list([expression])
    variables = standardize_symbolic_expression(variable_names)
    if len(variables) > 0:
        # Peel the outmost activation:
        function_name_list, _ = get_function_name_list(expression)
        if len(function_name_list) > 0:
            expression = expression.args[0]
        poly = Poly(expression, *variables)
        return poly.coeffs(), variable_names
    else:
        return [expression], []


def get_coeffs_tree(exprs, param_dict):
    """Get snapped coefficients by traversing the whole expression tree."""
    snapped_list = []
    length = 0
    for expr in exprs:
        length += get_coeffs_recur(expr, param_dict, snapped_list)
    return length, snapped_list


def get_coeffs_recur(expr, param_dict, snapped_list):
    import sympy

    if isinstance(expr, sympy.numbers.Float) or isinstance(expr, sympy.numbers.Integer):
        snapped_list.append(float(expr))
        return 1
    elif isinstance(expr, sympy.symbol.Symbol):
        if not expr.name.startswith("x"):
            if expr.name not in param_dict:
                raise Exception(
                    "Non-snapped parameter did not appear in param_dict! Check implementation."
                )
        return 1
    else:
        length = 0
        for sub_expr in expr.args:
            length += get_coeffs_recur(sub_expr, param_dict, snapped_list)
        length += 1
        return length


def standardize_symbolic_expression(symbolic_expression):
    """Standardize symbolic expression to be a list of SymPy expressions"""
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.utilities.lambdify import implemented_function

    if isinstance(symbolic_expression, str):
        symbolic_expression = parse_expr(symbolic_expression)
    if not (
        isinstance(symbolic_expression, list) or isinstance(symbolic_expression, tuple)
    ):
        symbolic_expression = [symbolic_expression]
    parsed_symbolic_expression = []
    for expression in symbolic_expression:
        parsed_expression = (
            parse_expr(expression) if isinstance(expression, str) else expression
        )
        if (
            hasattr(parsed_expression.func, "name")
            and parsed_expression.func.name in ACTIVATION_LIST
        ):
            activation = parsed_expression.func.name
            f = implemented_function(activation, get_activation(activation))
            parsed_expression = f(*parsed_expression.args)
        parsed_symbolic_expression.append(parsed_expression)
    return parsed_symbolic_expression


def get_number_DL(n, status):
    #     def rank(n):
    #         assert isinstance(n, int)
    #         if n == 0:
    #             return 1
    #         else:
    #             return 2 * abs(n) + int((1 - np.sign(n)) / 2)
    epsilon = 1e-10
    n = float(n)
    if status == "snapped":
        if np.abs(n - int(n)) < epsilon:
            return np.log2(1 + abs(int(n)))
        else:
            snapped = snap_core([n], "rational")[0][1]
            if snapped is not None and abs(n - snapped) < epsilon:
                _, numerator, denominator, _ = bestApproximation(n, 100)
                return np.log2((1 + abs(numerator)) * abs(denominator))
            else:
                if (n - np.pi) < 1e-10:
                    # It is pi:
                    return np.log2((1 + 3))
                elif (n - np.e) < 1e-10:
                    # It is e:
                    return np.log2((1 + 2))
                else:
                    raise Exception(
                        "The snapped numbers should be a rational number! However, {} is not a rational number.".format(
                            n
                        )
                    )
    elif status == "non-snapped":
        return np.log2(1 + (float(n) / PrecisionFloorLoss) ** 2) / 2
    else:
        raise Exception("status {0} not valid!".format(status))


def get_list_DL(List, status):
    if not (
        isinstance(List, list)
        or (
            isinstance(List, np.ndarray)
            and (len(List.shape) > 1 or (len(List.shape) == 1 and List.shape[0] > 1))
        )
    ):
        return get_number_DL(List, status)
    else:
        return np.sum([get_list_DL(element, status) for element in List])


def get_model_DL(model):
    if not (isinstance(model, list) or isinstance(model, tuple)):
        model = [model]
    return np.sum([model_ele.DL for model_ele in model])


def zero_grad_hook(idx):
    def hook_function(grad):
        grad[idx] = 0
        return grad

    return hook_function


## The following are snap functions for finding a best approximated integer or rational number for a real number:
def bestApproximation(x, imax):
    # The input is a numpy parameter vector p.
    # The output is an integer specifying which parameter to change,
    # and a float specifying the new value.
    def float2contfrac(x, nmax):
        c = [np.floor(x)]
        y = x - np.floor(x)
        k = 0
        while np.abs(y) != 0 and k < nmax:
            y = 1 / float(y)
            i = np.floor(y)
            c.append(i)
            y = y - i
            k = k + 1
        return c

    def contfrac2frac(seq):
        """ Convert the simple continued fraction in `seq` 
            into a fraction, num / den
        """
        num, den = 1, 0
        for u in reversed(seq):
            num, den = den + num * u, num
        return num, den

    def contFracRationalApproximations(c):
        return np.array(list(contfrac2frac(c[: i + 1]) for i in range(len(c))))

    def contFracApproximations(c):
        q = contFracRationalApproximations(c)
        return q[:, 0] / float(q[:, 1])

    def truncateContFrac(q, imax):
        k = 0
        while k < len(q) and np.maximum(np.abs(q[k, 0]), q[k, 1]) <= imax:
            k = k + 1
        return q[:k]

    def pval(p):
        return 1 - np.exp(-(p ** 0.87) / 0.36)

    xsign = np.sign(x)
    q = truncateContFrac(
        contFracRationalApproximations(float2contfrac(abs(x), 20)), imax
    )

    if len(q) > 0:
        p = (
            np.abs(q[:, 0] / q[:, 1] - abs(x)).astype(float)
            * (1 + np.abs(q[:, 0]))
            * q[:, 1]
        )
        p = pval(p)
        i = np.argmin(p)
        return (xsign * q[i, 0] / float(q[i, 1]), xsign * q[i, 0], q[i, 1], p[i])
    else:
        return (None, 0, 0, 1)


def integerSnap(p, top=1):
    metric = np.abs(p - np.round(p))
    chosen = np.argsort(metric)[:top]
    return list(zip(chosen, np.round(p)[chosen]))


def zeroSnap(p, top=1):
    metric = np.abs(p)
    chosen = np.argsort(metric)[:top]
    return list(zip(chosen, np.zeros(len(chosen))))


def rationalSnap(p, top=1):
    """Snap to nearest rational number using continued fraction."""
    snaps = np.array(list(bestApproximation(x, 100) for x in p))
    chosen = np.argsort(snaps[:, 3])[:top]
    return list(zip(chosen, snaps[chosen, 0]))


def vectorSnap(param_dict, top=1):
    """Divide p by its smallest number, and then perform integer snap.

    Args:
        param_dict: dictionary of parameter name and its values for snapping.
        top: number of parameters to snap.

    Returns:
        param_dict_subs: dictionary for substitution.
        new_param_dict: initial value for the remaining parameters.
    """
    tiny = 0.01
    huge = 1000000.0
    print(param_dict)
    p = list(param_dict.values())
    symbs = list(param_dict.keys())
    ap = np.abs(p)
    for i in range(len(p)):
        if ap[i] < tiny:
            ap[i] = huge
    i = np.argmin(ap)
    apmin = ap[i] * np.sign(p[i])
    if apmin >= huge:
        return {}, None
    else:
        q = p / apmin.astype(float)
        snap_targets = integerSnap(q, top=top + 1)
        param_dict_subs = {}
        new_param_dict = {}
        newsymb = symbs[i]
        for k in range(len(snap_targets)):
            if snap_targets[k][0] != i:
                if snap_targets[k][1] != 1:
                    param_dict_subs[symbs[snap_targets[k][0]]] = (
                        "{}".format(snap_targets[k][1]) + "*" + newsymb
                    )
                else:
                    param_dict_subs[symbs[snap_targets[k][0]]] = newsymb
        new_param_dict[newsymb] = param_dict[newsymb]
        return param_dict_subs, new_param_dict


def pairSnap(p, snap_mode="integer"):
    p = np.array(p)
    n = p.shape[0]
    pairs = []
    for i in range(1, n):
        for j in range(i):
            pairs.append([i, j, p[i] / p[j]])
            pairs.append([j, i, p[j] / p[i]])
    pairs = np.array(pairs)
    if snap_mode == "integer":
        (k, ratio) = integerSnap(pairs[:, 2])
    elif snap_mode == "rational":
        (k, ratio) = rationalSnap(pairs[:, 2])
    else:
        raise Exception("snap_mode {0} not recognized!".format(snap_mode))
    return (int(pairs[k, 0]), int(pairs[k, 1])), int(ratio)


# Finds best separable approximation of a matrix as M=ab^t.
# There's a degeneracy between rescaling a and rescaling b, which we fix by setting the smallest non-zero element of a equal to 1.
def separableSnap(M):
    (U, w, V) = np.linalg.svd(M)
    Mnew = w[0] * np.matmul(U[:, :1], V[:1, :])
    a = U[:, 0] * w[0]
    b = V[0, :]
    tiny = 0.001
    huge = 1000000.0
    aa = np.abs(a)
    for i in range(len(aa)):
        if aa[i] < tiny:
            aa[i] = huge
    i = np.argmin(aa)
    aamin = aa[i] * np.sign(aa[i])
    aa = aa / aamin
    return (a / aamin, b * aamin)


def snap_core(p, snap_mode, top=1):
    if len(p) == 0:
        return []
    else:
        if snap_mode == "zero":
            return zeroSnap(p, top=top)
        elif snap_mode == "integer":
            return integerSnap(p, top=top)
        elif snap_mode == "rational":
            return rationalSnap(p, top=top)
        elif snap_mode == "vector":
            return vectorSnap(p, top=top)
        elif snap_mode == "pair_integer":
            return pairSnap(p, snap_mode="integer")
        elif snap_mode == "pair_rational":
            return pairSnap(p, snap_mode="rational")
        elif snap_mode == "separable":
            return separableSnap(p)
        else:
            raise Exception("Snap mode {0} not recognized!".format(snap_mode))


def snap(param, snap_mode, excluded_idx=None, top=1):
    if excluded_idx is None or len(excluded_idx) == 0:
        return snap_core(param, snap_mode=snap_mode, top=top)
    else:
        full_idx = list(range(len(param)))
        valid_idx = sorted(list(set(full_idx) - set(excluded_idx)))
        valid_dict = list(enumerate(valid_idx))
        param_valid = [param[i] for i in valid_idx]
        snap_targets_valid = snap_core(param_valid, snap_mode=snap_mode, top=top)
        snap_targets = []
        for idx_valid, new_value in snap_targets_valid:
            snap_targets.append((valid_dict[idx_valid][1], new_value))
        return snap_targets


def unsnap(exprs, param_dict):
    """Unsnap a symbolic expression, tranforming all numerical values to learnable parameters."""
    unsnapped_param_dict = {}
    unsnapped_exprs = []
    for expr in exprs:
        unsnapped_expr = unsnap_recur(expr, param_dict, unsnapped_param_dict)
        unsnapped_exprs.append(unsnapped_expr)
    return unsnapped_exprs, unsnapped_param_dict


def unsnap_recur(expr, param_dict, unsnapped_param_dict):
    """Recursively transform each numerical value into a learnable parameter."""
    import sympy
    from sympy import Symbol

    if isinstance(expr, sympy.numbers.Float) or isinstance(expr, sympy.numbers.Integer):
        used_param_names = list(param_dict.keys()) + list(unsnapped_param_dict)
        unsnapped_param_name = get_next_available_key(
            used_param_names, "p", is_underscore=False
        )
        unsnapped_param_dict[unsnapped_param_name] = float(expr)
        unsnapped_expr = Symbol(unsnapped_param_name)
        return unsnapped_expr
    elif isinstance(expr, sympy.symbol.Symbol):
        return expr
    else:
        unsnapped_sub_expr_list = []
        for sub_expr in expr.args:
            unsnapped_sub_expr = unsnap_recur(
                sub_expr, param_dict, unsnapped_param_dict
            )
            unsnapped_sub_expr_list.append(unsnapped_sub_expr)
        return expr.func(*unsnapped_sub_expr_list)


def get_next_available_key(
    iterable, key, midfix="", suffix="", is_underscore=True, start_from_null=False
):
    """Get the next available key that does not collide with the keys in the dictionary."""
    if start_from_null and key + suffix not in iterable:
        return key + suffix
    else:
        i = 0
        underscore = "_" if is_underscore else ""
        while "{}{}{}{}{}".format(key, underscore, midfix, i, suffix) in iterable:
            i += 1
        new_key = "{}{}{}{}{}".format(key, underscore, midfix, i, suffix)
        return new_key


def update_dictionary(dictionary, key, item):
    """Update the key: item in the dictionary. If key collision happens, 
    rename the key if the items do not refer to the same thing."""
    if key in dictionary:
        if dictionary[key] is item:
            result = 2
            return result, key
        else:
            result = 0
            new_key = get_next_available_key(dictionary, key)
            dictionary[new_key] = item
            return result, new_key
    else:
        result = 1
        dictionary[key] = item
        return result, key


def pprint_dict(dictionary):
    string = ""
    for key, item in dictionary.items():
        string += "{}: {}; ".format(key, item)
    string = string[:-2]
    return string


class Batch_Generator(object):
    def __init__(self, X, y, batch_size=50, target_one_hot_off=False):
        """
        Initilize the Batch_Generator class
        """
        if isinstance(X, tuple):
            self.binary_X = True
        else:
            self.binary_X = False
        if isinstance(X, Variable):
            X = X.cpu().data.numpy()
        if isinstance(y, Variable):
            y = y.data.numpy()
        self.target_one_hot_off = target_one_hot_off

        if self.binary_X:
            X1, X2 = X
            self.sample_length = len(y)
            assert (
                len(X1) == len(X2) == self.sample_length
            ), "X and y must have the same length!"
            assert (
                batch_size <= self.sample_length
            ), "batch_size must not be larger than \
                    the number of samples!"
            self.batch_size = int(batch_size)

            X1_len = [element.shape[-1] for element in X1]
            X2_len = [element.shape[-1] for element in X2]
            y_len = [element.shape[-1] for element in y]
            if len(np.unique(X1_len)) == 1 and len(np.unique(X2_len)) == 1:
                assert (
                    len(np.unique(y_len)) == 1
                ), "when X1 and X2 has only one size, the y should also have only one size!"
                self.combine_size = False
                self.X1 = np.array(X1)
                self.X2 = np.array(X2)
                self.y = np.array(y)
                self.index = np.array(range(self.sample_length))
                self.idx_batch = 0
                self.idx_epoch = 0
            else:
                self.combine_size = True
                self.X1 = X1
                self.X2 = X2
                self.y = y
                self.input_dims_list = zip(X1_len, X2_len)
                self.input_dims_dict = {}
                for i, input_dims in enumerate(self.input_dims_list):
                    if input_dims not in self.input_dims_dict:
                        self.input_dims_dict[input_dims] = {"idx": [i]}
                    else:
                        self.input_dims_dict[input_dims]["idx"].append(i)
                for input_dims in self.input_dims_dict:
                    idx = np.array(self.input_dims_dict[input_dims]["idx"])
                    self.input_dims_dict[input_dims]["idx"] = idx
                    self.input_dims_dict[input_dims]["X1"] = [X1[i] for i in idx]
                    self.input_dims_dict[input_dims]["X2"] = [X2[i] for i in idx]
                    self.input_dims_dict[input_dims]["y"] = [y[i] for i in idx]
                    self.input_dims_dict[input_dims]["idx_batch"] = 0
                    self.input_dims_dict[input_dims]["idx_epoch"] = 0
                    self.input_dims_dict[input_dims]["index"] = np.array(
                        range(len(idx))
                    ).astype(int)
        else:
            self.sample_length = len(y)
            assert (
                batch_size <= self.sample_length
            ), "batch_size must not be larger than \
                    the number of samples!"
            self.batch_size = int(batch_size)

            X_len = [element.shape[-1] for element in X]
            y_len = [element.shape[-1] for element in y]

            if len(np.unique(X_len)) == 1 and len(np.unique(y_len)) == 1:
                assert (
                    len(np.unique(y_len)) == 1
                ), "when X has only one size, the y should also have only one size!"
                self.combine_size = False
                self.X = np.array(X)
                self.y = np.array(y)
                self.index = np.array(range(self.sample_length))
                self.idx_batch = 0
                self.idx_epoch = 0
            else:
                self.combine_size = True
                self.X = X
                self.y = y
                self.input_dims_list = zip(X_len, y_len)
                self.input_dims_dict = {}
                for i, input_dims in enumerate(self.input_dims_list):
                    if input_dims not in self.input_dims_dict:
                        self.input_dims_dict[input_dims] = {"idx": [i]}
                    else:
                        self.input_dims_dict[input_dims]["idx"].append(i)
                for input_dims in self.input_dims_dict:
                    idx = np.array(self.input_dims_dict[input_dims]["idx"])
                    self.input_dims_dict[input_dims]["idx"] = idx
                    self.input_dims_dict[input_dims]["X"] = [X[i] for i in idx]
                    self.input_dims_dict[input_dims]["y"] = [y[i] for i in idx]
                    self.input_dims_dict[input_dims]["idx_batch"] = 0
                    self.input_dims_dict[input_dims]["idx_epoch"] = 0
                    self.input_dims_dict[input_dims]["index"] = np.array(
                        range(len(idx))
                    ).astype(int)

    def reset(self):
        """Reset the index and batch iteration to the initialization state"""
        if not self.combine_size:
            self.index = np.array(range(self.sample_length))
            self.idx_batch = 0
            self.idx_epoch = 0
        else:
            for input_dims in self.input_dims_dict:
                self.input_dims_dict[input_dims]["idx_batch"] = 0
                self.input_dims_dict[input_dims]["idx_epoch"] = 0
                self.input_dims_dict[input_dims]["index"] = np.array(
                    range(len(idx))
                ).astype(int)

    def next_batch(self, mode="random", isTorch=False, is_cuda=False, given_dims=None):
        """Generate each batch with the same size (even if the examples and target may have variable size)"""
        if self.binary_X:
            if not self.combine_size:
                start = self.idx_batch * self.batch_size
                end = (self.idx_batch + 1) * self.batch_size

                if end > self.sample_length:
                    self.idx_epoch += 1
                    self.idx_batch = 0
                    np.random.shuffle(self.index)
                    start = 0
                    end = self.batch_size

                self.idx_batch += 1
                chosen_index = self.index[start:end]
                y_batch = deepcopy(self.y[chosen_index])
                if self.target_one_hot_off:
                    y_batch = y_batch.argmax(-1)
                X1_batch = deepcopy(self.X1[chosen_index])
                X2_batch = deepcopy(self.X2[chosen_index])
            else:  # If the input_dims have variable size
                if mode == "random":
                    if given_dims is None:
                        rand = np.random.choice(self.sample_length)
                        input_dims = self.input_dims_list[rand]
                    else:
                        input_dims = given_dims
                    length = len(self.input_dims_dict[input_dims]["idx"])
                    if self.batch_size >= length:
                        chosen_index = np.random.choice(
                            length, size=self.batch_size, replace=True
                        )
                    else:
                        start = (
                            self.input_dims_dict[input_dims]["idx_batch"]
                            * self.batch_size
                        )
                        end = (
                            self.input_dims_dict[input_dims]["idx_batch"] + 1
                        ) * self.batch_size
                        if end > length:
                            self.input_dims_dict[input_dims]["idx_epoch"] += 1
                            self.input_dims_dict[input_dims]["idx_batch"] = 0
                            np.random.shuffle(self.input_dims_dict[input_dims]["index"])
                            start = 0
                            end = self.batch_size
                        self.input_dims_dict[input_dims]["idx_batch"] += 1
                        chosen_index = self.input_dims_dict[input_dims]["index"][
                            start:end
                        ]
                    y_batch = deepcopy(
                        np.array(
                            [
                                self.input_dims_dict[input_dims]["y"][j]
                                for j in chosen_index
                            ]
                        )
                    )
                    if self.target_one_hot_off:
                        y_batch = y_batch.argmax(-1)
                    X1_batch = deepcopy(
                        np.array(
                            [
                                self.input_dims_dict[input_dims]["X1"][j]
                                for j in chosen_index
                            ]
                        )
                    )
                    X2_batch = deepcopy(
                        np.array(
                            [
                                self.input_dims_dict[input_dims]["X2"][j]
                                for j in chosen_index
                            ]
                        )
                    )

            if isTorch:
                X1_batch = Variable(
                    torch.from_numpy(X1_batch), requires_grad=False
                ).type(torch.FloatTensor)
                X2_batch = Variable(
                    torch.from_numpy(X2_batch), requires_grad=False
                ).type(torch.FloatTensor)
                if self.target_one_hot_off:
                    y_batch = Variable(
                        torch.from_numpy(y_batch), requires_grad=False
                    ).type(torch.LongTensor)
                else:
                    y_batch = Variable(
                        torch.from_numpy(y_batch), requires_grad=False
                    ).type(torch.FloatTensor)

                if is_cuda:
                    X1_batch = X1_batch.cuda()
                    X2_batch = X2_batch.cuda()
                    y_batch = y_batch.cuda()

            return (X1_batch, X2_batch), y_batch
        else:
            if not self.combine_size:
                start = self.idx_batch * self.batch_size
                end = (self.idx_batch + 1) * self.batch_size

                if end > self.sample_length:
                    self.idx_epoch += 1
                    self.idx_batch = 0
                    np.random.shuffle(self.index)
                    start = 0
                    end = self.batch_size

                self.idx_batch += 1
                chosen_index = self.index[start:end]
                y_batch = deepcopy(self.y[chosen_index])
                if self.target_one_hot_off:
                    y_batch = y_batch.argmax(-1)
                X_batch = deepcopy(self.X[chosen_index])
            else:  # If the input_dims have variable size
                if mode == "random":
                    if given_dims is None:
                        rand = np.random.choice(self.sample_length)
                        input_dims = self.input_dims_list[rand]
                    else:
                        input_dims = given_dims
                    length = len(self.input_dims_dict[input_dims]["idx"])
                    if self.batch_size >= length:
                        chosen_index = np.random.choice(
                            length, size=self.batch_size, replace=True
                        )
                    else:
                        start = (
                            self.input_dims_dict[input_dims]["idx_batch"]
                            * self.batch_size
                        )
                        end = (
                            self.input_dims_dict[input_dims]["idx_batch"] + 1
                        ) * self.batch_size
                        if end > length:
                            self.input_dims_dict[input_dims]["idx_epoch"] += 1
                            self.input_dims_dict[input_dims]["idx_batch"] = 0
                            np.random.shuffle(self.input_dims_dict[input_dims]["index"])
                            start = 0
                            end = self.batch_size
                        self.input_dims_dict[input_dims]["idx_batch"] += 1
                        chosen_index = self.input_dims_dict[input_dims]["index"][
                            start:end
                        ]
                    y_batch = deepcopy(
                        np.array(
                            [
                                self.input_dims_dict[input_dims]["y"][j]
                                for j in chosen_index
                            ]
                        )
                    )
                    if self.target_one_hot_off:
                        y_batch = y_batch.argmax(-1)
                    X_batch = deepcopy(
                        np.array(
                            [
                                self.input_dims_dict[input_dims]["X"][j]
                                for j in chosen_index
                            ]
                        )
                    )
            if isTorch:
                X_batch = Variable(torch.from_numpy(X_batch), requires_grad=False).type(
                    torch.FloatTensor
                )
                if self.target_one_hot_off:
                    y_batch = Variable(
                        torch.from_numpy(y_batch), requires_grad=False
                    ).type(torch.LongTensor)
                else:
                    y_batch = Variable(
                        torch.from_numpy(y_batch), requires_grad=False
                    ).type(torch.FloatTensor)

                if is_cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

            return X_batch, y_batch


def logmeanexp(tensor, axis, keepdims=False):
    """Calculate logmeanexp of tensor."""
    return torch.logsumexp(tensor, axis=axis, keepdims=keepdims) - np.log(
        tensor.shape[axis]
    )


def str2bool(v):
    """used for argparse, 'type=str2bool', so that can pass in string True or False."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true"):
        return True
    elif v.lower() in ("false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def compute_class_correspondence(predicted_domain, true_domain, verbose=True):
    """Compute the best match of two lists of labels among all permutations."""

    def mismatches(domain1, domain2):
        return (domain1 != domain2).sum()

    def associate(domain, current_ids, new_ids):
        new_domain = deepcopy(domain)
        for i, current_id in enumerate(current_ids):
            new_domain[domain == current_id] = new_ids[i]
        return new_domain

    assert len(predicted_domain.shape) == len(true_domain.shape) == 1
    assert isinstance(predicted_domain, np.ndarray)
    assert isinstance(true_domain, np.ndarray)
    predicted_domain = predicted_domain.flatten().astype(int)
    true_domain = true_domain.flatten().astype(int)
    predicted_ids = np.unique(predicted_domain)
    true_ids = np.unique(true_domain)

    union_ids = np.sort(list(set(predicted_ids).union(set(true_ids))))
    if len(union_ids) > 10:
        if verbose:
            print("num_domains = {0}, too large!".format(len(union_ids)))
        return None, None, None
    min_num_mismatches = np.inf
    argmin_permute = None
    predicted_domain_argmin = None

    for i, union_ids_permute in enumerate(itertools.permutations(union_ids)):
        predicted_domain_permute = associate(
            predicted_domain, union_ids, union_ids_permute
        )
        num_mismatches = mismatches(predicted_domain_permute, true_domain)
        if num_mismatches < min_num_mismatches:
            min_num_mismatches = num_mismatches
            argmin_permute = union_ids_permute
            predicted_domain_argmin = predicted_domain_permute
        if verbose and i % 100000 == 0:
            print(i)
        if min_num_mismatches == 0:
            break
    return (
        predicted_domain_argmin,
        min_num_mismatches,
        list(zip(union_ids, argmin_permute)),
    )


def compute_class_correspondence_greedy(
    predicted_domain, true_domain, verbose=True, threshold=7
):
    """Compute the best match of two lists of labels among all permutations; if number of labels
        is greater than the threshold, the labels with top threshold + 1 numbers perform its own permutation.
    """

    def mismatches(domain1, domain2):
        return (domain1 != domain2).sum()

    def associate(domain, current_ids, new_ids):
        new_domain = deepcopy(domain)
        for i, current_id in enumerate(current_ids):
            new_domain[domain == current_id] = new_ids[i]
        return new_domain

    assert len(predicted_domain.shape) == len(true_domain.shape) == 1
    assert isinstance(predicted_domain, np.ndarray)
    assert isinstance(true_domain, np.ndarray)
    predicted_domain = predicted_domain.flatten().astype(int)
    true_domain = true_domain.flatten().astype(int)
    predicted_ids, _ = get_sorted_counts(predicted_domain)
    true_ids, _ = get_sorted_counts(true_domain)
    assert len(predicted_ids) == len(true_ids)

    if len(true_ids) > threshold:
        num_greedy_ids = max(2, len(true_ids) - threshold)
    else:
        num_greedy_ids = -1

    min_num_mismatches = np.inf
    argmin_permute = None
    predicted_domain_argmin = None

    total = 0
    for i, ids_permute_greedy in enumerate(
        itertools.permutations(true_ids[: num_greedy_ids + 1])
    ):
        ids_inner = ids_permute_greedy[-1:] + tuple(true_ids[num_greedy_ids + 1 :])
        for j, ids_permute_all in enumerate(itertools.permutations(ids_inner)):
            permuted_id = ids_permute_greedy[:-1] + ids_permute_all
            predicted_domain_permute = associate(
                predicted_domain, predicted_ids, permuted_id
            )
            try:
                num_mismatches = mismatches(predicted_domain_permute, true_domain)
            except:
                import pdb

                pdb.set_trace()
            if num_mismatches < min_num_mismatches:
                min_num_mismatches = num_mismatches
                argmin_permute = (predicted_ids, permuted_id)
                predicted_domain_argmin = predicted_domain_permute
            if verbose and total % 100000 == 0:
                print(total)
            total += 1
            if min_num_mismatches == 0:
                break
    return predicted_domain_argmin, min_num_mismatches, list(zip(*argmin_permute))


def acc_spectral_clutering(
    X,
    y,
    n_neighbors=10,
    num_runs=1,
    affinity="nearest_neighbors",
    matching_method="full_permute",
    **kwargs
):
    X, y = to_np_array(X, y)
    n_clusters = len(np.unique(y))
    n_components = n_clusters

    acc_list = []
    for i in range(num_runs):
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            n_components=n_components,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
        ).fit(X)
        cluster_labels = clustering.labels_

        if matching_method == "full_permute":
            _, min_num_mismatches, _ = compute_class_correspondence(
                cluster_labels, y, **kwargs
            )
        elif matching_method == "greedy_permute":
            _, min_num_mismatches, _ = compute_class_correspondence_greedy(
                cluster_labels, y, **kwargs
            )
        else:
            raise
        acc = 1 - min_num_mismatches / float(len(y))
        acc_list.append(acc)
    return np.max(acc_list), cluster_labels


def get_sorted_counts(array):
    array_unique, counts = np.unique(array, return_counts=True)
    counts, array_unique = sort_two_lists(counts, array_unique, reverse=True)
    return array_unique, counts


def filter_kwargs(kwargs, param_names):
    new_kwargs = {}
    for key, item in kwargs.items():
        if key in param_names:
            new_kwargs[key] = item
    return new_kwargs


def get_key_of_largest_value(Dict):
    return max(Dict.items(), key=operator.itemgetter(1))[0]


def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def compose_two_keylists(keys1, keys2):
    """Return a fully broadcast list of keys, combining keys1 and keys2."""
    length1 = [len(key) for key in keys1]
    length2 = [len(key) for key in keys2]
    assert len(np.unique(length1)) <= 1
    assert len(np.unique(length2)) <= 1
    if len(length1) == 0:
        return keys2
    if len(length2) == 0:
        return keys1
    length1 = length1[0]
    length2 = length2[0]

    # Find the maximum length where both keys share the same sub_keys:
    is_same = True
    for i in range(1, min(length1, length2) + 1):
        sub_keys1 = [key[:i] for key in keys1]
        sub_keys2 = [key[:i] for key in keys2]
        if set(sub_keys1) != set(sub_keys2):
            assert len(set(sub_keys1).intersection(set(sub_keys2))) == 0
            is_same = False
            break
    if is_same:
        largest_common_length = i
    else:
        largest_common_length = i - 1

    new_keys = []
    keys_common = remove_duplicates([key[:largest_common_length] for key in keys1])
    keys1_reduced = remove_duplicates([key[largest_common_length:] for key in keys1])
    keys2_reduced = remove_duplicates([key[largest_common_length:] for key in keys2])
    for key_common in keys_common:
        if length1 > largest_common_length:
            for key1 in keys1_reduced:
                if length2 > largest_common_length:
                    for key2 in keys2_reduced:
                        key = key_common + key1 + key2
                        new_keys.append(key)
                else:
                    key = key_common + key1
                    new_keys.append(key)

        else:
            if length2 > largest_common_length:
                for key2 in keys2_reduced:
                    key = key_common + key2
                    new_keys.append(key)
            else:
                key = key_common
                new_keys.append(key)
    return new_keys


def broadcast_keys(key_list_all):
    """Return a fully broadcast {new_broadcast_key: list of Arg keys}

    key_list_all: a list of items, where each item is a list of keys.
    For example, key_list_all = [[(0, "s"), (0, "d"), (1, "d)], [0, 1], None], it will return:

    key_dict = {(0, "s"): [(0, "s"), 0, None],
                (0, "d"): [(0, "d"), 0, None],
                (1, "d"): [(1, "d"), 1, None]}
    Here None denotes that there is only one input, which will be broadcast to all.
    """
    # First: get all the combinations
    new_key_list = []
    for i, keys in enumerate(key_list_all):
        if keys is not None:
            keys = [(ele,) if not isinstance(ele, tuple) else ele for ele in keys]
            new_key_list = compose_two_keylists(new_key_list, keys)

    ## new_key_list: a list of fully broadcast keys
    ## key_list_all: a list of original_key_list, each of which corresponds to
    ##               the keys of an OrderedDict()
    key_dict = {}
    for new_key in new_key_list:
        new_key_map_list = []
        is_match_all = True
        for original_key_list in key_list_all:
            if original_key_list is None:
                new_key_map_list.append(None)
                is_match = True
            else:
                is_match = False
                for key in original_key_list:
                    key = (key,) if not isinstance(key, tuple) else key
                    if set(key).issubset(set(new_key)):
                        is_match = True
                        if len(key) == 1:
                            key = key[0]
                        new_key_map_list.append(key)
                        break
            is_match_all = is_match_all and is_match
        if is_match_all:
            if len(new_key) == 1:
                new_key = new_key[0]
            key_dict[new_key] = new_key_map_list
    return key_dict


def split_bucket(dictionary, num_common):
    """Split the dictionary into multiple buckets, determined by key[num_common:]."""
    from multiset import Multiset

    keys_common = remove_duplicates([key[:num_common] for key in dictionary.keys()])
    # Find the different keys:
    keys_diff = []
    for key in dictionary.keys():
        if Multiset(keys_common[0]).issubset(Multiset(key)):
            if key[num_common:] not in keys_diff:
                keys_diff.append(key[num_common:])
    keys_diff = sorted(keys_diff)

    buckets = [OrderedDict() for _ in range(len(keys_diff))]
    for key, item in dictionary.items():
        id = keys_diff.index(key[num_common:])
        buckets[id][key[:num_common]] = item
    return buckets


def get_list_elements(List, string_idx):
    """Select elements of the list based on string_idx.
    
    Format of string_idx:
        if starting with "r", means first performs random permutation.
        "100:200": the 100th to 199th elements
        "100:" : the 100th elements and onward
        ":200" : the 0th to 199th elements
        "150" : the 150th element
        "::" : all elements
    """
    # Permute if starting with "r":
    if string_idx.startswith("r"):
        List = np.random.permutation(List).tolist()
        string_idx = string_idx[1:]
    # Select indices:
    if string_idx == "::":
        return List
    elif ":" in string_idx:
        string_split = string_idx.split(":")
        string_split = [string for string in string_split if len(string) != 0]
        if len(string_split) == 2:
            start_idx, end_idx = string_idx.split(":")
            start_idx, end_idx = eval(start_idx), eval(end_idx)
            if end_idx > len(List):
                raise Exception("The end index exceeds the length of the list!")
            list_selected = List[start_idx:end_idx]
        elif len(string_split) == 1:
            if string_idx.startswith(":"):
                list_selected = List[: eval(string_idx[1:])]
            else:
                list_selected = List[eval(string_idx[:-1]) :]
        else:
            raise
    else:
        string_idx = eval(string_idx)
        list_selected = [List[string_idx]]
    return list_selected


def get_generalized_mean(List, cumu_mode="mean", epsilon=1e-10):
    """Get generalized-mean of elements in the list"""
    List = np.array(list(List))
    assert len(List.shape) == 1

    if cumu_mode[0] == "gm" and cumu_mode[1] == 1:
        cumu_mode = "mean"
    elif cumu_mode[0] == "gm" and cumu_mode[1] == 0:
        cumu_mode = "geometric"
    elif cumu_mode[0] == "gm" and cumu_mode[1] == -1:
        cumu_mode = "harmonic"

    # Obtain mean:
    if cumu_mode == "mean":
        mean = List.mean()
    elif cumu_mode == "min":
        mean = np.min(List)
    elif cumu_mode == "max":
        mean = np.max(List)
    elif cumu_mode == "harmonic":
        mean = len(List) / (1 / (List + epsilon)).sum()
    elif cumu_mode == "geometric":
        mean = (List + epsilon).prod() ** (1 / float(len(List)))
    elif cumu_mode[0] == "gm":
        order = cumu_mode[1]
        mean = (np.minimum((List + epsilon) ** order, 1e30).mean()) ** (
            1 / float(order)
        )
    else:
        raise
    return mean


def upper_first(string):
    """Return a new string with the first letter capitalized."""
    if len(string) > 0:
        return string[0].upper() + string[1:]
    else:
        return string


def lower_first(string):
    """Return a new string with the first letter capitalized."""
    if len(string) > 0:
        return string[0].lower() + string[1:]
    else:
        return string


def update_dict(Dict, key, value):
    """Return a new dictionary with the item with key updated by the corresponding value"""
    if Dict is None:
        return None
    new_dict = deepcopy(Dict)
    new_dict[key] = value
    return new_dict


def get_root_dir(suffix):
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index(suffix)
    dirname = "/".join(dirname_split[: index + 1])
    return dirname


def check_same_set(List_of_List):
    """Return True if each element list has the same set of elements."""
    if len(List_of_List) == 0:
        return None
    List = List_of_List[0]
    for List_ele in List_of_List:
        if set(List) != set(List_ele):
            return False
    return True


def check_same_dict(Dict, value_list, key_list):
    """Check if the value stored is the same as the newly given value_list.
    Return a list of keys whose values are different from the stored ones.
    """
    if len(Dict) == 0:
        for key, value in zip(key_list, value_list):
            Dict[key] = value
        return []
    else:
        not_equal_list = []
        for key, value in zip(key_list, value_list):
            value_stored = Dict[key]
            if (
                isinstance(value, Number)
                or isinstance(value, tuple)
                or isinstance(value, list)
            ):
                is_equal = value == value_stored
                if not is_equal:
                    not_equal_list.append(key)
            else:
                if tuple(value.shape) != tuple(value_stored.shape):
                    not_equal_list.append(key)
                else:
                    is_equal = (value == value_stored).all()
                    if not is_equal:
                        not_equal_list.append(key)
        return not_equal_list


def print_banner(string, banner_size=100):
    """Pring the string sandwidched by two lines."""
    print("\n" + "=" * banner_size + "\n" + string + "\n" + "=" * banner_size + "\n")
