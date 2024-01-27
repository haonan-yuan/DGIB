"""
    Robust Graph Convolutional Networks Against Adversarial Attacks. KDD 2019.
        http://pengcui.thumedialab.com/papers/RGCN.pdf
    Author's Tensorflow implemention:
        https://github.com/thumanlab/nrlweb/tree/master/static/assets/download
"""

import torch.nn.functional as F
import math
import pdb
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.distributions.multivariate_normal import MultivariateNormal
from DGIB.DeepRobust.deeprobust.graph import utils
import torch.optim as optim
from copy import deepcopy

# TODO sparse implementation


class GGCL_F(Module):
    """GGCL: the input is feature"""

    def __init__(self, in_features, out_features, dropout=0.6):
        super(GGCL_F, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, features, adj_norm1, adj_norm2, gamma=1):
        features = F.dropout(features, self.dropout, training=self.training)
        self.miu = F.elu(torch.mm(features, self.weight_miu))
        self.sigma = F.relu(torch.mm(features, self.weight_sigma))
        # torch.mm(previous_sigma, self.weight_sigma)
        Att = torch.exp(-gamma * self.sigma)
        miu_out = adj_norm1 @ (self.miu * Att)
        sigma_out = adj_norm2 @ (self.sigma * Att * Att)
        return miu_out, sigma_out


class GGCL_D(Module):

    """GGCL_D: the input is distribution"""

    def __init__(self, in_features, out_features, dropout):
        super(GGCL_D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        # self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, miu, sigma, adj_norm1, adj_norm2, gamma=1):
        miu = F.dropout(miu, self.dropout, training=self.training)
        sigma = F.dropout(sigma, self.dropout, training=self.training)
        self.miu = F.elu(miu @ self.weight_miu)
        self.sigma = F.relu(sigma @ self.weight_sigma)

        Att = torch.exp(-gamma * self.sigma)
        mean_out = adj_norm1 @ (self.miu * Att)
        sigma_out = adj_norm2 @ (self.sigma * Att * Att)
        return mean_out, sigma_out


class GaussianConvolution(Module):
    def __init__(self, in_features, out_features):
        super(GaussianConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        # self.sigma = Parameter(torch.FloatTensor(out_features))
        # self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # TODO
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(
        self, previous_miu, previous_sigma, adj_norm1=None, adj_norm2=None, gamma=1
    ):

        if adj_norm1 is None and adj_norm2 is None:
            return (
                torch.mm(previous_miu, self.weight_miu),
                torch.mm(previous_miu, self.weight_miu),
            )
            # torch.mm(previous_sigma, self.weight_sigma)

        Att = torch.exp(-gamma * previous_sigma)
        M = adj_norm1 @ (previous_miu * Att) @ self.weight_miu
        Sigma = adj_norm2 @ (previous_sigma * Att * Att) @ self.weight_sigma
        return M, Sigma

        # M = torch.mm(torch.mm(adj, previous_miu * A), self.weight_miu)
        # Sigma = torch.mm(torch.mm(adj, previous_sigma * A * A), self.weight_sigma)

        # TODO sparse implemention
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        # return output + self.bias

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class RGCN(Module):
    def __init__(
        self,
        nnodes,
        nfeat,
        nhid,
        nclass,
        gamma=1.0,
        beta1=5e-4,
        beta2=5e-4,
        lr=0.01,
        dropout=0.6,
        device="cpu",
        num_layers=2,
    ):
        super(RGCN, self).__init__()

        self.device = device
        # adj_norm = normalize(adj)
        # first turn original features to distribution
        self.lr = lr
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.nclass = nclass
        self.nhid = nhid // 2
        # self.gc1 = GaussianConvolution(nfeat, nhid, dropout=dropout)
        # self.gc2 = GaussianConvolution(nhid, nclass, dropout)
        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(
                self,
                "gc{}".format(i + 1),
                GGCL_F(nfeat, nhid, dropout=dropout)
                if i == 0
                else GGCL_D(
                    nhid, nclass if i == num_layers - 1 else nhid, dropout=dropout
                ),
            )
            # self.gc1 = GGCL_F(nfeat, nhid, dropout=dropout)
            # self.gc2 = GGCL_D(nhid, nclass, dropout=dropout)

        self.dropout = dropout
        # self.gaussian = MultivariateNormal(torch.zeros(self.nclass), torch.eye(self.nclass))
        self.gaussian = MultivariateNormal(
            torch.zeros(nnodes, self.nclass),
            torch.diag_embed(torch.ones(nnodes, self.nclass)),
        )
        self.adj_norm1, self.adj_norm2 = None, None
        self.features, self.labels = None, None

    def forward(self):
        features = self.features
        for i in range(self.num_layers):
            if i == 0:
                miu, sigma = getattr(self, "gc{}".format(i + 1))(
                    features, self.adj_norm1, self.adj_norm2, self.gamma
                )
            else:
                miu, sigma = getattr(self, "gc{}".format(i + 1))(
                    miu, sigma, self.adj_norm1, self.adj_norm2, self.gamma
                )
        # miu, sigma = self.gc1(features, self.adj_norm1, self.adj_norm2, self.gamma)
        # miu, sigma = self.gc2(miu, sigma, self.adj_norm1, self.adj_norm2, self.gamma)

        sigma_last = getattr(self, "gc{}".format(self.num_layers)).sigma
        output = miu + self.gaussian.sample().to(self.device) * torch.sqrt(
            sigma_last + 1e-8
        )
        return F.log_softmax(output, dim=1)

    def fit(
        self,
        features,
        adj,
        labels,
        idx_train,
        idx_val=None,
        train_iters=200,
        verbose=True,
    ):

        adj, features, labels = utils.to_tensor(
            adj.todense(), features.todense(), labels, device=self.device
        )

        self.features, self.labels = features, labels
        self.adj_norm1 = self._normalize_adj(adj, power=-1 / 2)
        self.adj_norm2 = self._normalize_adj(adj, power=-1)
        print("=== training rgcn model ===")
        self._initialize()
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print("Epoch {}, training loss: {}".format(i, loss_train.item()))

        self.eval()
        output = self.forward()
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print("Epoch {}, training loss: {}".format(i, loss_train.item()))

            self.eval()
            output = self.forward()
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                #                 print ("step", i)
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        print(
            "=== picking the best model according to the performance on validation ==="
        )
        self.load_state_dict(weights)

    def test(self, idx_test):
        # output = self.forward()
        output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print(
            "Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),
        )
        return [loss_test.item(), acc_test.item()]

    def _loss(self, input, labels):
        loss = F.nll_loss(input, labels)
        miu1 = getattr(self, "gc%d" % (self.num_layers - 1)).miu
        sigma1 = getattr(self, "gc%d" % (self.num_layers - 1)).sigma
        # miu1 = self.gc1.miu
        # sigma1 = self.gc1.sigma
        kl_loss = 0.5 * (miu1.pow(2) + sigma1 - torch.log(1e-8 + sigma1)).mean(1)
        kl_loss = kl_loss.sum()
        norm2 = torch.norm(
            getattr(self, "gc%d" % (self.num_layers - 1)).weight_miu, 2
        ).pow(2) + torch.norm(
            getattr(self, "gc%d" % (self.num_layers - 1)).weight_sigma, 2
        ).pow(
            2
        )
        # norm2 = torch.norm(self.gc1.weight_miu, 2).pow(2) + \
        #         torch.norm(self.gc1.weight_sigma, 2).pow(2)

        # print(f'gcn_loss: {loss.item()}, kl_loss: {self.beta1 * kl_loss.item()}, norm2: {self.beta2 * norm2.item()}')
        return loss + self.beta1 * kl_loss + self.beta2 * norm2

    def _initialize(self):
        for i in range(self.num_layers):
            getattr(self, "gc%d" % (i + 1)).reset_parameters()
        # self.gc1.reset_parameters()
        # self.gc2.reset_parameters()

    def _normalize_adj(self, adj, power=-1 / 2):

        """Row-normalize sparse matrix"""
        A = adj + torch.eye(len(adj)).to(self.device)
        D_power = (A.sum(1)).pow(power)
        D_power[torch.isinf(D_power)] = 0.0
        D_power = torch.diag(D_power)
        return D_power @ A @ D_power

    def predict(self, features=None, adj=None):
        """By default, inputs are unnormalized data"""

        self.eval()
        if features is None and adj is None:
            return self.forward()
        else:
            if type(adj) is not torch.Tensor:
                adj, features = utils.to_tensor(
                    adj.todense(), features.todense(), device=self.device
                )

            self.features = features
            self.adj_norm1 = self._normalize_adj(adj, power=-1 / 2)
            self.adj_norm2 = self._normalize_adj(adj, power=-1)
            return self.forward()
