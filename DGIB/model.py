from torch_geometric.nn.inits import glorot, zeros
from torch import nn
from torch.nn import Parameter
from torch_geometric.utils import (
    add_remaining_self_loops,
    remove_self_loops,
    add_self_loops,
    softmax,
    degree,
    to_undirected,
)
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn.conv import MessagePassing
from torch.distributions.normal import Normal
from torch_scatter import scatter
from collections import OrderedDict
import torch.nn.functional as F
import networkx as nx
import numpy as np
import torch
import math

from DGIB.pytorch_net.net import reparameterize, Mixture_Gaussian_reparam
from DGIB.pytorch_net.util import (
    sample,
    to_cpu_recur,
    to_np_array,
    to_Variable,
    record_data,
    make_dir,
    remove_duplicates,
    update_dict,
    get_list_elements,
    to_string,
    filter_filename,
)
from DGIB.util_IB import (
    get_reparam_num_neurons,
    sample_lognormal,
    scatter_sample,
    uniform_prior,
    compose_log,
    edge_index_2_csr,
    COLOR_LIST,
    LINESTYLE_LIST,
    process_data_for_nettack,
    parse_filename,
    add_distant_neighbors,
)


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=50, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()

        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = torch.cat([x_i, x_j], dim=1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).squeeze()


class MultiplyPredictor(torch.nn.Module):
    def __init__(self):
        super(MultiplyPredictor, self).__init__()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = (x_i * x_j).sum(dim=1)
        return torch.sigmoid(x)


class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()

        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias


class DGIBConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        reparam_mode=None,
        prior_mode=None,
        distribution=None,
        temperature=0.2,
        nbsz=15,
        val_use_mean=True,
        sample_size=1,
        bias=True,
        agg_param=0.8,
        **kwargs
    ):
        super(DGIBConv, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.reparam_mode = reparam_mode if reparam_mode != "None" else None
        self.prior_mode = prior_mode
        self.out_neurons = out_channels
        self.distribution = distribution
        self.temperature = temperature
        self.nbsz = nbsz
        self.sample_size = sample_size
        self.val_use_mean = val_use_mean
        self.agg_param = agg_param
        self.weight = Parameter(torch.Tensor(in_channels, heads * self.out_neurons))
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_neurons))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * self.out_neurons))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(self.out_neurons))
        else:
            self.register_parameter("bias", None)

        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(
                    is_reparam=False,
                    Z_size=self.out_channels,
                    n_components=n_components,
                )

        self.skip_editing_edge_index = False
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward_single_graphshot(self, x, edge_index, size=None):
        if size is None and torch.is_tensor(x) and not self.skip_editing_edge_index:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (
                None if x[0] is None else torch.matmul(x[0], self.weight),
                None if x[1] is None else torch.matmul(x[1], self.weight),
            )

        out = self.propagate(edge_index, size=size, x=x)

        if self.reparam_mode is not None:
            # Reparameterize:
            out = out.view(-1, self.out_neurons)
            self.dist, _ = reparameterize(
                model=None, input=out, mode=self.reparam_mode, size=self.out_channels,
            )  # dist: [B * head, Z]
            Z_core = sample(self.dist, self.sample_size)  # [S, B * head, Z]
            Z = Z_core.view(
                self.sample_size, -1, self.heads * self.out_channels
            )  # [S, B, head * Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(
                    loc=torch.zeros(out.size(0), self.out_channels).to(x.device),
                    scale=torch.ones(out.size(0), self.out_channels).to(x.device),
                )  # feature_prior: [B * head, Z]

            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = (
                    torch.distributions.kl.kl_divergence(self.dist, self.feature_prior)
                    .sum(-1)
                    .view(-1, self.heads)
                    .mean(-1)
                )
            else:
                Z_logit = (
                    self.dist.log_prob(Z_core).sum(-1)
                    if self.reparam_mode.startswith("diag")
                    else self.dist.log_prob(Z_core)
                )  # [S, B * head]
                prior_logit = self.feature_prior.log_prob(Z_core).sum(
                    -1
                )  # [S, B * head]
                # upper bound of I(X; Z):
                ixz = (
                    (Z_logit - prior_logit).mean(0).view(-1, self.heads).mean(-1)
                )  # [B]

            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)
            else:
                out = (
                    out[:, : self.out_channels]
                    .contiguous()
                    .view(-1, self.heads * self.out_channels)
                )
        else:
            ixz = torch.zeros(x.size(0)).to(x.device)

        if "categorical" in self.distribution:
            structure_kl_loss = torch.sum(
                self.alpha * torch.log((self.alpha + 1e-16) / self.prior)
            )
        elif "Bernoulli" in self.distribution:
            posterior = torch.distributions.bernoulli.Bernoulli(self.alpha)
            prior = torch.distributions.bernoulli.Bernoulli(logits=self.prior)
            structure_kl_loss = (
                torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()
            )
        else:
            raise Exception(
                "I think this belongs to the diff subset sampling that is not implemented"
            )

        return out, torch.mean(ixz), structure_kl_loss

    def forward(self, x_all, edge_index_all, size=None):
        times = len(x_all)
        dev = x_all[0].device
        n = len(x_all[0])
        out_list = torch.zeros((times, x_all[0].size(0), self.out_channels)).to(dev)
        self.ixz_list = [[] for _ in range(times)]
        self.structure_kl_loss_list = [[] for _ in range(times)]

        for t in range(times):
            (
                out,
                self.ixz_list[t],
                self.structure_kl_loss_list[t],
            ) = self.forward_single_graphshot(x_all[t], edge_index_all[t])
            if t > 0:
                weights = F.sigmoid(
                    torch.tensor(list(range(t))).view(t, 1, 1).to(x_all[0].device)
                )
                out_list[t] = (
                    torch.sum(weights * out_list[:t], dim=0) / t
                ) * self.agg_param + out * (1 - self.agg_param)
            else:
                out_list[t] = out
        ixz_mean = torch.mean(torch.stack(self.ixz_list))
        structure_kl_loss_mean = torch.mean(torch.stack(self.structure_kl_loss_list))

        return out_list, ixz_mean, structure_kl_loss_mean, self.ixz_list[t - 1]

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(
            -1, self.heads, self.out_neurons
        )  # [N_edge, heads, out_channels]
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_neurons :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_neurons)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(
                dim=-1
            )  # [N_edge, heads]

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Sample attention coefficients stochastically.
        neighbor_sampling_mode = self.distribution
        if "categorical" in neighbor_sampling_mode:
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
            self.alpha = alpha
            self.prior = uniform_prior(edge_index_i)
            if self.val_use_mean is False or self.training:
                temperature = self.temperature
                sample_neighbor_size = self.nbsz
                if neighbor_sampling_mode == "categorical":
                    alpha = scatter_sample(alpha, edge_index_i, temperature, size_i)
                elif "multi-categorical" in neighbor_sampling_mode:
                    alphas = []
                    for _ in range(
                        sample_neighbor_size
                    ):  #! this can be improved by parallel sampling
                        alphas.append(
                            scatter_sample(alpha, edge_index_i, temperature, size_i)
                        )
                    alphas = torch.stack(alphas, dim=0)
                    if "sum" in neighbor_sampling_mode:
                        alpha = alphas.sum(dim=0)
                    elif "max" in neighbor_sampling_mode:
                        alpha, _ = torch.max(alphas, dim=0)
                    else:
                        raise
                else:
                    raise
        elif neighbor_sampling_mode == "Bernoulli":
            alpha_normalization = torch.ones_like(alpha)
            alpha_normalization = softmax(
                alpha_normalization, edge_index_i, num_nodes=size_i
            )
            alpha = torch.clamp(torch.sigmoid(alpha), 0.01, 0.99)
            self.alpha = alpha
            self.prior = (torch.ones_like(self.alpha) * (1 / self.nbsz)).to(
                alpha.device
            )
            if not self.val_use_mean or self.training:
                temperature = self.temperature
                alpha = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                    torch.Tensor([temperature]).to(alpha.device), probs=alpha
                ).rsample()
            alpha = alpha * alpha_normalization
        else:
            raise

        return (x_j * alpha.view(-1, self.heads, 1)).view(
            -1, self.heads * self.out_neurons
        )

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_neurons)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def to_device(self, device):
        self.to(device)
        return self

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class DGIBNN(torch.nn.Module):
    def __init__(self, args=None):

        super(DGIBNN, self).__init__()
        self.num_features = args.nfeat
        self.normalize = True
        self.reparam_mode = args.reparam_mode
        self.prior_mode = args.prior_mode  #
        self.distribution = args.distribution
        self.temperature = args.temperature
        self.nbsz = args.nbsz
        self.dropout = args.dropout
        self.latent_size = args.nhid
        self.sample_size = args.sample_size
        self.num_layers = args.n_layers
        self.with_relu = True
        self.val_use_mean = True
        self.reparam_all_layers = True
        self.is_cuda = True
        self.device = torch.device(
            self.is_cuda
            if isinstance(self.is_cuda, str)
            else "cuda"
            if self.is_cuda
            else "cpu"
        )
        self.ixz = 0
        self.structure_kl_loss = 0
        self.consensual = 0
        self.use_RTE = args.use_RTE
        self.edge_decoder = MultiplyPredictor()
        self.args = args

        self.init()

    def time_encoding(self, x_all):
        n_hid = self.latent_size
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

        if self.use_RTE:
            times = len(x_all)
            for t in range(times):
                x_all[t] = x_all[t] + self.lin(
                    self.emb(torch.LongTensor([t]).to(x_all[t].device))
                )
        return x_all

    def init(self):
        self.reparam_layers = []
        latent_size = self.latent_size

        for i in range(self.num_layers):
            if i == 0:
                input_size = self.num_features
            else:
                input_size = latent_size
            if self.reparam_all_layers is True:
                is_reparam = True
            elif isinstance(self.reparam_all_layers, tuple):
                reparam_all_layers = tuple(
                    [
                        kk + self.num_layers if kk < 0 else kk
                        for kk in self.reparam_all_layers
                    ]
                )
                is_reparam = i in reparam_all_layers
            else:
                raise
            if is_reparam:
                self.reparam_layers.append(i)
            setattr(
                self,
                "conv{}".format(i + 1),
                DGIBConv(
                    input_size,
                    latent_size,
                    heads=self.args.heads if i != self.num_layers - 1 else 1,
                    concat=True,
                    reparam_mode=self.reparam_mode if is_reparam else None,
                    prior_mode=self.prior_mode if is_reparam else None,
                    distribution=self.distribution,
                    temperature=self.temperature,
                    nbsz=self.nbsz,
                    val_use_mean=self.val_use_mean,
                    sample_size=self.sample_size,
                ),
            )

        self.reparam_layers = sorted(self.reparam_layers)

        self.reg_params = self.parameters()
        self.non_reg_params = OrderedDict()  ##
        self.to(self.device)

    def to_device(self, device):
        for i in range(self.num_layers):
            getattr(self, "conv{}".format(i + 1)).to_device(device)
        self.to(device)
        return self

    def forward(self, x_all, edge_index_all):
        times = len(x_all)
        ixz, structure_kl_loss, consensual = [], [], []
        for i in range(self.num_layers):
            layer = getattr(self, "conv{}".format(i + 1))
            x_all, ixz_mean, structure_kl_loss_mean, ixz_cons = layer(
                x_all, edge_index_all
            )
            ixz.append(ixz_mean)
            structure_kl_loss.append(structure_kl_loss_mean)
            consensual.append(ixz_cons)

            x_all = [F.elu(x) for x in x_all]
            x_all = [
                F.dropout(
                    input=F.normalize(x, dim=-1), p=self.dropout, training=self.training
                )
                for x in x_all
            ]
        return (
            x_all,
            torch.stack(ixz).mean(),
            torch.stack(structure_kl_loss).mean(),
            torch.stack(consensual).mean(),
        )
