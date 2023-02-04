from torch_geometric.utils import to_undirected, remove_self_loops
from sklearn.metrics import roc_auc_score
import torch.nn.functional as fn
import torch.optim as optim
import torch.nn as nn
import torch.autograd
import torch
import numpy as np
import tempfile
import random

from dataset import load_nc_dataset

from model.DSF_GPR_I import DSF_GPR_I
from model.DSF_GPR_R import DSF_GPR_R
from model.DSF_Bern_I import DSF_Bern_I
from model.DSF_Bern_R import DSF_Bern_R
from model.DSF_Jacobi import DSF_Jacobi_I, DSF_Jacobi_R


def log_print(args, str):
    if not args.hpm_opt_mode:
        print(str)


def eval_acc(targ, prob):
    # generalized version for both single/multi-label classification
    pred = prob.max(dim=-1)[1].type_as(targ)
    acc = pred.eq(targ.squeeze(dim=-1)).double().sum() / targ.numel()
    acc = acc.item()
    return acc


def eval_rocauc(y_true, y_pred):
    """
    adopted from
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py
    https://github.com/CUAI/Non-Homophily-Benchmarks/blob/main/data_utils.py
    """
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = fn.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


def keep2decimal(x):
    return format(x, ".2f")


def get_rnd_seed():
    # return a random seed in 6 numbers
    return np.random.randint(99999, 999999)


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_model(args, c, d, device):
    if args.method == "DSF_GPR_R":
        model = DSF_GPR_R(d, args.hidden_channels, c, args, args.pe_indim, args.pe_hid_dim, args.PE_alpha).to(device)
    elif args.method == "DSF_GPR_I":
        model = DSF_GPR_I(d, args.hidden_channels, c, args, args.pe_indim, args.pe_hid_dim, args.PE_alpha, args.PE_beta).to(device)
    elif args.method == "DSF_Bern_R":
        model = DSF_Bern_R(d, args.hidden_channels, c, args, args.pe_indim, args.pe_hid_dim, args.PE_alpha).to(device)
    elif args.method == "DSF_Bern_I":
        model = DSF_Bern_I(d, args.hidden_channels, c, args, args.pe_indim, args.pe_hid_dim, args.PE_alpha, args.PE_beta).to(device)
    elif args.method == "DSF_Jacobi_R":
        model = DSF_Jacobi_R(args.alpha, args.dpb, args.dpt, d, c, sole=args.mysole,
                             PE_dropout=args.PE_dropout, PE_in_dim=args.pe_indim, PE_hid_dim=args.pe_hid_dim,
                             PE_alpha=args.PE_alpha,
                             a=args.a, b=args.b).to(device)
    elif args.method == "DSF_Jacobi_I":
        model = DSF_Jacobi_I(args.alpha, args.dpb, args.dpt, d, c, sole=args.mysole,
                             PE_dropout=args.PE_dropout, PE_in_dim=args.pe_indim, PE_hid_dim=args.pe_hid_dim,
                             PE_alpha=args.PE_alpha, PE_beta=args.PE_beta,
                             a=args.a, b=args.b).to(device)
    else:
        raise ValueError('Invalid method')
    return model


class PE_reg_loss(nn.Module):
    def __init__(self, edge_index, pe_hid_dim, dev, pe_normalize=False):
        super(PE_reg_loss, self).__init__()
        self.edge_index = edge_index
        self.pe_hid_dim = pe_hid_dim
        self.identity_mat = torch.eye(pe_hid_dim).to(dev)
        self.pe_normalize = pe_normalize

    def forward(self, pe):
        if self.pe_normalize:
            pe = pe - torch.mean(pe, dim=0, keepdim=True)
            pe = fn.normalize(pe, dim=0, p=2)
        # ort-loss
        if self.pe_normalize:
            PTP_In = torch.matmul(pe.T, pe) - self.identity_mat
        else:
            PTP_In = torch.matmul(pe.T, pe).fill_diagonal_(0)
        pe_ort_loss = torch.pow(torch.norm(PTP_In, "fro"), 2) / self.pe_hid_dim
        return pe_ort_loss


class EvalHelper:
    def __init__(self, args):
        use_cuda = torch.cuda.is_available() and not args.cpu
        dev = torch.device('cuda' if use_cuda else 'cpu')

        # load data
        dataset = load_nc_dataset(args.dataset, args.sub_dataset)
        dataset.graph["edge_index"] = remove_self_loops(dataset.graph["edge_index"])[0]
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        dataset.graph["pos_enc"] = torch.load(f"{args.DATAPATH}node_pos_enc/{args.pos_enc_type}_{args.dataset}_indim_{args.pe_indim}.pt")
        c = dataset.label.max().item() + 1
        d = dataset.graph['node_feat'].shape[1]

        log_print(args, "Load a random split: trn/val/tst=60%/20%/20%")
        split_dic = np.load(f"{args.DATAPATH}{args.dataset}_{args.sub_dataset}_randomSplit.npy", allow_pickle=True).item()
        trn_idx, val_idx, tst_idx = split_dic["trn_idx"], split_dic["val_idx"], split_dic["tst_idx"]
        assert len(set(trn_idx).intersection(val_idx)) == 0
        assert len(set(trn_idx).intersection(tst_idx)) == 0
        assert len(set(val_idx).intersection(tst_idx)) == 0

        # data to cuda
        trn_idx = torch.from_numpy(trn_idx).to(dev)
        val_idx = torch.from_numpy(val_idx).to(dev)
        tst_idx = torch.from_numpy(tst_idx).to(dev)
        if len(dataset.label.shape) == 1:
            dataset.label = dataset.label.unsqueeze(1)
        dataset.label = dataset.label.to(dev)
        dataset.graph['edge_index'] = dataset.graph['edge_index'].to(dev)
        dataset.graph['node_feat'] = dataset.graph['node_feat'].to(dev)
        dataset.graph['pos_enc'] = dataset.graph['pos_enc'].to(dev)

        # model configs
        _, dsf_bse, dsf_mode = args.method.split("_")
        assert dsf_bse in ["GPR", "Bern", "Jacobi"]
        assert dsf_mode in ["I", "R"]
        model = parse_model(args, c, d, dev)
        pe_reg_loss = PE_reg_loss(dataset.graph['edge_index'], args.pe_hid_dim, dev, args.pe_normalize).to(dev)
        if dsf_bse == "Jacobi":
            optmz = optim.Adam([{
                'params': model.emb.parameters(),
                'weight_decay': args.wd1,
                'lr': args.lr1
            }, {
                'params': model.comb.parameters(),
                'weight_decay': args.wd3,
                'lr': args.lr3
            }, {
                'params': model.pe_agg.parameters(),
                'weight_decay': args.wd4,
                'lr': args.lr4
            }])
        else:
            all_params = model.parameters()
            optmz = optim.Adam(all_params, lr=args.lr, weight_decay=args.reg)
        # bce/rocauc as the loss/eval function for twitch-e dataset
        loss_fn = nn.BCEWithLogitsLoss() if args.dataset == 'twitch-e' else nn.NLLLoss()
        eval_fn = eval_rocauc if args.dataset == 'twitch-e' else eval_acc

        self.dataset = dataset
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.model, self.optmz = model, optmz
        self.dsf_bse, self.dsf_mode = dsf_bse, dsf_mode
        self.loss_fn, self.eval_fn = loss_fn, eval_fn
        self.pe_reg_loss = pe_reg_loss
        self.args = args

    def before_loss(self, args, out):
        if args.dataset == 'twitch-e':
            true_label = fn.one_hot(self.dataset.label, self.dataset.label.max() + 1).type(out.dtype)
        else:
            true_label = self.dataset.label
            out = fn.log_softmax(out, dim=1)
        return out, true_label

    def run_epoch(self, args):
        self.model.train()
        self.optmz.zero_grad()
        if self.dsf_bse == "Jacobi":
            out, pe = self.model(self.dataset.graph['node_feat'], self.dataset.graph['edge_index'], torch.ones_like(self.dataset.graph['edge_index'][0]), pe=self.dataset.graph['pos_enc'])
        else:
            out, pe = self.model(self.dataset)
        out, true_label = self.before_loss(args, out)
        task_loss = self.loss_fn(out[self.trn_idx], true_label.squeeze(1)[self.trn_idx])
        if self.dsf_mode == "R":
            pe_ort_loss = self.pe_reg_loss(pe)
            loss = task_loss + args.ort_pe_lambda * pe_ort_loss
        else:
            loss = task_loss
        loss.backward()
        self.optmz.step()
        if self.dsf_mode == "R":
            log_print(args, "epoch-loss={:.4f}, task-loss={:.4f}, pe-orth-loss={:.4f}".format(loss.item(), task_loss.item(), pe_ort_loss.item()))
        else:
            log_print(args, "epoch-loss={:.4f}".format(loss.item()))
        return loss.item()

    def evaluate(self):
        self.model.eval()
        if self.dsf_bse == "Jacobi":
            out, _ = self.model(self.dataset.graph['node_feat'], self.dataset.graph['edge_index'], torch.ones_like(self.dataset.graph['edge_index'][0]), pe=self.dataset.graph['pos_enc'])
        else:
            out, _ = self.model(self.dataset)
        trn_acc = self.eval_fn(self.dataset.label[self.trn_idx], out[self.trn_idx])
        val_acc = self.eval_fn(self.dataset.label[self.val_idx], out[self.val_idx])
        tst_acc = self.eval_fn(self.dataset.label[self.tst_idx], out[self.tst_idx])
        return trn_acc, val_acc, tst_acc


def model_run(args):
    # random initialization
    set_rng_seed(args.rnd_seed)
    # build model
    agent = EvalHelper(args)
    # model-training
    wait_cnt = 0
    best_val_acc = 0.0
    best_model_sav = tempfile.TemporaryFile()
    for t in range(args.nepoch):
        agent.run_epoch(args)
        trn_acc, val_acc, tst_acc = agent.evaluate()
        log_print(args, "epoch: {}/{}, trn-acc={:.4f}%, val-acc={:.4f}%, tst-acc={:.4f}%".format(
            t + 1, args.nepoch, trn_acc * 100, val_acc * 100, tst_acc * 100))
        # early-stop
        if val_acc > best_val_acc:
            wait_cnt = 0
            best_val_acc = val_acc
            best_model_sav.close()
            best_model_sav = tempfile.TemporaryFile()
            torch.save(agent.model.state_dict(), best_model_sav)
        else:
            wait_cnt += 1
            if wait_cnt > args.early:
                break
    # final results
    log_print(args, "Load selected model ...")
    best_model_sav.seek(0)
    agent.model.load_state_dict(torch.load(best_model_sav))
    trn_acc, val_acc, tst_acc = agent.evaluate()
    return val_acc, tst_acc


def DSF(args):
    args.rnd_seed = get_rnd_seed()
    val_acc, tst_acc = model_run(args)
    log_print(args, f"{args.method} evaluated on {args.dataset}_{args.sub_dataset}:"
                    f" val-acc={keep2decimal(val_acc * 100)}%, tst-acc={keep2decimal(tst_acc * 100)}%")


def config(args):
    ########################################################################
    args.method = "DSF_GPR_R"
    args.dataset = "chameleon"
    args.pos_enc_type = "RW_PE"
    args.pe_normalize = False
    args.reg = 3e-7
    args.lr = 0.05
    args.dropout = 0.3
    args.dprate = 0.7
    args.PE_dropout = 0
    args.PE_alpha = 0.9
    args.pe_indim = 24
    args.ort_pe_lambda = 0.5
    args.Init = "PPR"
    args.alpha = 0.9
    ########################################################################
    # args.method = "DSF_GPR_I"
    # args.dataset = "chameleon"
    # args.pos_enc_type = "LAP_PE"
    # args.reg = 3e-7
    # args.lr = 0.07
    # args.dropout = 0.2
    # args.dprate = 0.7
    # args.PE_dropout = 0.4
    # args.PE_alpha = 1
    # args.pe_indim = 30
    # args.Init = "NPPR"
    # args.alpha = 0.2
    # args.PE_beta = 0.5
    ########################################################################
    # args.method = "DSF_Bern_R"
    # args.dataset = "chameleon"
    # args.pos_enc_type = "RW_PE"
    # args.pe_normalize = False
    # args.reg = 8e-7
    # args.lr = 0.07
    # args.dropout = 0.2
    # args.dprate = 0.7
    # args.PE_dropout = 0
    # args.PE_alpha = 0.6
    # args.pe_indim = 2
    # args.ort_pe_lambda = 0.6
    ########################################################################
    # args.method = "DSF_Bern_I"
    # args.dataset = "chameleon"
    # args.pos_enc_type = "LAP_PE"
    # args.reg = 1e-7
    # args.lr = 0.09
    # args.dropout = 0.1
    # args.dprate = 0.7
    # args.PE_dropout = 0.6
    # args.PE_alpha = 1
    # args.pe_indim = 14
    # args.PE_beta = 0.2
    ########################################################################
    # args.method = "DSF_Jacobi_R"
    # args.dataset = "chameleon"
    # args.pos_enc_type = "RW_PE"
    # args.mysole = True
    # args.pe_normalize = True
    # args.ort_pe_lambda = 1.0
    # args.PE_dropout = 0.1
    # args.PE_alpha = 0.5
    # args.pe_indim = 24
    # args.lr1 = 0.09
    # args.lr3 = 0.02
    # args.lr4 = 0.02
    # args.wd1 = 4e-7
    # args.wd3 = 9e-6
    # args.wd4 = 7e-5
    # args.alpha = 1.0
    # args.a = 0
    # args.b = 0.5
    # args.dpb = 0.7
    # args.dpt = 0.1
    ########################################################################
    # args.method = "DSF_Jacobi_I"
    # args.dataset = "chameleon"
    # args.pos_enc_type = "LAP_PE"
    # args.mysole = True
    # args.PE_dropout = 0.1
    # args.PE_alpha = 0.5
    # args.PE_beta = 0.8
    # args.pe_indim = 30
    # args.lr1 = 0.06
    # args.lr3 = 0.06
    # args.lr4 = 0.06
    # args.wd1 = 9e-8
    # args.wd3 = 4e-7
    # args.wd4 = 1e-3
    # args.alpha = 1
    # args.a = 0.2
    # args.b = 0.5
    # args.dpb = 0.6
    # args.dpt = 0.2
    ########################################################################
    return args


def main():
    from parse import args
    DSF(config(args))


if __name__ == '__main__':
    main()
