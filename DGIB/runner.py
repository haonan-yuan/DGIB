from DGIB.utils.mutils import *
from DGIB.utils.inits import prepare
from DGIB.utils.loss import EnvLoss
from DGIB.utils.util import init_logger, logger
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import pandas as pd
import math
import wandb


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class Runner(object):
    def __init__(self, args, model, data, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.data = data
        self.model = model
        self.writer = writer
        self.len = len(data["train"]["edge_index_list"])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        self.nbsz = args.nbsz
        self.in_dim = args.nfeat
        self.hid_dim = args.nhid

        if len(data["x"]) == self.len:
            x = [
                self.data["x"][t].to(args.device).clone().detach()
                for t in range(self.len)
            ]
            self.x = x
        else:
            x = data["x"].to(args.device).clone().detach()
            self.x = [x for _ in range(self.len)]
        self.edge_index_list_pre = [
            data["train"]["edge_index_list"][ix].long().to(args.device)
            for ix in range(self.len)
        ]

        self.loss = EnvLoss(args)
        print("total length: {}, test length: {}".format(self.len, args.testlength))

    def loss_cvae(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss

    def train(self, epoch, data):
        args = self.args
        self.model.train()
        optimizer = self.optimizer

        embeddings, ixz_loss, structure_kl_loss, consensual_loss = self.model(
            self.x,
            [
                data["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
        )
        device = embeddings[0].device

        val_auc_list = []
        test_auc_list = []
        train_auc_list = []
        for t in range(self.len - 1):
            z = embeddings[t]
            _, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        edge_index = []
        pos_edge_index_all = []
        neg_edge_index_all = []
        edge_label = []
        tsize = []
        for t in range(self.len_train - 1):
            z = embeddings[t]
            pos_edge_index = prepare(data, t + 1)[0]
            if args.dataset == "yelp":
                neg_edge_index = bi_negative_sampling(
                    pos_edge_index, args.num_nodes, args.shift
                )
            else:
                neg_edge_index = negative_sampling(
                    pos_edge_index,
                    num_neg_samples=pos_edge_index.size(1) * args.sampling_times,
                )
            edge_index.append(torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            pos_edge_index_all.append(pos_edge_index)
            neg_edge_index_all.append(neg_edge_index)
            pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
            neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
            edge_label.append(torch.cat([pos_y, neg_y], dim=0))
            tsize.append(pos_edge_index.shape[1] * 2)

        edge_label = torch.cat(edge_label, dim=0)

        def generate_edge_index_interv(pos_edge_index_all, neg_edge_index_all, indices):
            edge_label = []
            edge_index = []
            pos_edge_index_interv = pos_edge_index_all.copy()
            neg_edge_index_interv = neg_edge_index_all.copy()
            index = indices.cpu().numpy()
            for t in range(self.len_train - 1):
                mask_pos = np.logical_and(
                    np.isin(pos_edge_index_interv[t].cpu()[0], index),
                    np.isin(pos_edge_index_interv[t].cpu()[1], index),
                )
                pos_edge_index = pos_edge_index_interv[t][:, mask_pos]
                mask_neg = np.logical_and(
                    np.isin(neg_edge_index_interv[t].cpu()[0], index),
                    np.isin(neg_edge_index_interv[t].cpu()[1], index),
                )
                neg_edge_index = neg_edge_index_interv[t][:, mask_neg]
                pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
                neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
                edge_label.append(torch.cat([pos_y, neg_y], dim=0))
                edge_index.append(torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            edge_label = torch.cat(edge_label, dim=0)
            return edge_label, edge_index

        def cal_y(embeddings, decoder):
            preds = torch.tensor([]).to(device)
            for t in range(self.len_train - 1):
                z = embeddings[t]
                pred = decoder(z, edge_index[t])
                preds = torch.cat([preds, pred])
            return preds

        criterion = torch.nn.BCELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        pred_y = cal_y(embeddings, self.model.edge_decoder)
        main_loss = cal_loss(pred_y, edge_label)

        alpha = 1 - args.alpha
        beta_1 = args.beta
        beta_2 = args.beta

        loss = (
            main_loss
            + (1 - alpha) * consensual_loss
            + beta_1 * ixz_loss
            + beta_2 * structure_kl_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_epoch_loss = loss.item()

        return average_epoch_loss, train_auc_list, val_auc_list, test_auc_list

    def run(self):
        args = self.args
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0

        self.optimizer = optim.Adam(
            [p for n, p in self.model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        t_total0 = time.time()
        max_auc = 0
        max_test_auc = 0
        max_train_auc = 0

        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                epoch_losses, train_auc_list, val_auc_list, test_auc_list = self.train(
                    epoch, self.data["train"]
                )
                average_epoch_loss = epoch_losses
                average_train_auc = np.mean(train_auc_list)
                average_val_auc = np.mean(val_auc_list)
                average_test_auc = np.mean(test_auc_list)

                if average_val_auc > max_auc:
                    max_auc = average_val_auc
                    max_test_auc = average_test_auc
                    max_train_auc = average_train_auc

                    test_results = self.test(epoch, self.data["test"])

                    metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(
                        ","
                    )
                    measure_dict = dict(
                        zip(
                            metrics,
                            [max_train_auc, max_auc, max_test_auc] + test_results,
                        )
                    )

                    patience = 0

                    PATH = "../checkpoint/" + self.args.dataset + ".pth"
                    # PATH = os.path.join(wandb.run.dir, 'model.pth')
                    torch.save({"model_state_dict": self.model.state_dict()}, PATH)

                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print(
                        "Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(
                            epoch, average_epoch_loss, time.time() - t0
                        )
                    )
                    print(
                        f"Current: Epoch:{epoch}, Train AUC:{average_train_auc:.4f}, Val AUC: {average_val_auc:.4f}, Test AUC: {average_test_auc:.4f}"
                    )
                    print(
                        f"Train: Epoch:{test_results[0]}, Train AUC:{max_train_auc:.4f}, Val AUC: {max_auc:.4f}, Test AUC: {max_test_auc:.4f}"
                    )
                    print(
                        f"Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}"
                    )

        epoch_time = (time.time() - t_total0) / (epoch - 1)
        metrics = [max_train_auc, max_auc, max_test_auc] + test_results + [epoch_time]
        metrics_des = "clean_train_auc,clean_val_auc,clean_test_auc,epoch,att_struct_train_auc,att_struct_val_auc,att_struct_test_auc,epoch_time".split(
            ","
        )
        metrics_dict = dict(zip(metrics_des, metrics))
        df = pd.DataFrame([metrics], columns=metrics_des)
        print(df)
        return metrics_dict

    def test(self, epoch, data):
        args = self.args

        train_auc_list = []

        val_auc_list = []
        test_auc_list = []
        self.model.eval()
        embeddings, _, _, _ = self.model(
            self.x,
            [
                data["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
        )

        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        return [
            epoch,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list),
        ]

    def re_run(self):
        args = self.args
        data_clean = self.data["train"]
        data_attacked = self.data["test"]

        filepath = "../saved_model/" + self.args.dataset + ".pth"
        checkpoint = torch.load(filepath)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        clean_train_auc_list = []
        clean_val_auc_list = []
        clean_test_auc_list = []
        att_feat_05_train_auc_list = []
        att_feat_05_val_auc_list = []
        att_feat_05_test_auc_list = []
        att_feat_10_train_auc_list = []
        att_feat_10_val_auc_list = []
        att_feat_10_test_auc_list = []
        att_feat_15_train_auc_list = []
        att_feat_15_val_auc_list = []
        att_feat_15_test_auc_list = []
        att_struct_train_auc_list = []
        att_struct_val_auc_list = []
        att_struct_test_auc_list = []

        self.model.eval()
        embeddings, _, _, _ = self.model(
            self.x,
            [
                data_clean["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
        )

        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data_clean, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                clean_train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                clean_val_auc_list.append(auc)
            else:
                clean_test_auc_list.append(auc)

        embeddings, _, _, _ = self.model(
            self.x,
            [
                data_attacked["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
        )

        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data_attacked, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                att_struct_train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                att_struct_val_auc_list.append(auc)
            else:
                att_struct_test_auc_list.append(auc)

        x = self.data["x_noise_0.5"].to(args.device).clone().detach()
        self.x_noise = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        embeddings, _, _, _ = self.model(
            self.x_noise,
            [
                data_attacked["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
        )

        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data_attacked, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                att_feat_05_train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                att_feat_05_val_auc_list.append(auc)
            else:
                att_feat_05_test_auc_list.append(auc)

        x = self.data["x_noise_1.0"].to(args.device).clone().detach()
        self.x_noise = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        embeddings, _, _, _ = self.model(
            self.x_noise,
            [
                data_attacked["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
        )

        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data_attacked, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                att_feat_10_train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                att_feat_10_val_auc_list.append(auc)
            else:
                att_feat_10_test_auc_list.append(auc)

        x = self.data["x_noise_1.5"].to(args.device).clone().detach()
        self.x_noise = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        embeddings, _, _, _ = self.model(
            self.x_noise,
            [
                data_attacked["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
        )

        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data_attacked, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                att_feat_15_train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                att_feat_15_val_auc_list.append(auc)
            else:
                att_feat_15_test_auc_list.append(auc)

        test_res = [
            0,
            np.mean(clean_train_auc_list),
            np.mean(clean_val_auc_list),
            np.mean(clean_test_auc_list),
            np.mean(att_feat_05_test_auc_list),
            np.mean(att_feat_10_test_auc_list),
            np.mean(att_feat_15_test_auc_list),
            np.mean(att_struct_test_auc_list),
        ]

        metrics = "epoch,clean_train_auc,clean_val_auc,clean_test_auc,att_feat_05_test_auc,att_feat_10_test_auc,att_feat_15_test_auc,att_struct_test_auc".split(
            ","
        )
        metrics_dict = dict(zip(metrics, test_res))
        df = pd.DataFrame([test_res], columns=metrics)
        print(df)
        return metrics_dict

    def re_run_evasive(self):
        args = self.args
        data = self.data["train"]

        if self.args.distribution == "Bernoulli":
            filepath = (
                "../saved_model/original_evasive/bernoulli/"
                + self.args.dataset.split("_")[0]
                + ".pth"
            )
        elif self.args.distribution == "categorical":
            filepath = (
                "../saved_model/original_evasive/categorical/"
                + self.args.dataset.split("_")[0]
                + ".pth"
            )
        checkpoint = torch.load(filepath)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        train_auc_list = []
        val_auc_list = []
        test_auc_list = []

        x = [
            self.data["x"][t].to(args.device).clone().detach() for t in range(self.len)
        ]
        self.x = x
        embeddings, _, _, _ = self.model(
            self.x,
            [
                data["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
        )

        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        test_res = [
            0,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list),
        ]

        metrics = "epoch,train_auc,val_auc,test_auc".split(",")
        metrics_dict = dict(zip(metrics, test_res))
        df = pd.DataFrame([test_res], columns=metrics)
        print(df)
        return metrics_dict

    def re_run_poisoning(self):
        args = self.args
        data = self.data["train"]

        if self.args.distribution == "Bernoulli":
            filepath = (
                "../saved_model/poisoning/bernoulli/" + self.args.dataset + ".pth"
            )
        elif self.args.distribution == "categorical":
            filepath = (
                "../saved_model/poisoning/categorical/" + self.args.dataset + ".pth"
            )
        checkpoint = torch.load(filepath)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        train_auc_list = []
        val_auc_list = []
        test_auc_list = []

        x = [
            self.data["x"][t].to(args.device).clone().detach() for t in range(self.len)
        ]
        self.x = x
        embeddings, _, _, _ = self.model(
            self.x,
            [
                data["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
        )

        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        test_res = [
            0,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list),
        ]

        metrics = "epoch,train_auc,val_auc,test_auc".split(",")
        metrics_dict = dict(zip(metrics, test_res))
        df = pd.DataFrame([test_res], columns=metrics)
        print(df)
        return metrics_dict
