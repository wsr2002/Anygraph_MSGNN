import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch_geometric.transforms as T
from model import Feat_Projector, Adj_Projector, TopoEncoder
import os


class MultiDataHandler:
    def __init__(self, trn_datasets, tst_datasets_group):
        all_datasets = trn_datasets
        all_tst_datasets = []
        for tst_datasets in tst_datasets_group:
            all_datasets = all_datasets + tst_datasets
            all_tst_datasets += tst_datasets
        all_datasets = list(set(all_datasets))
        all_datasets.sort()
        self.trn_handlers = []
        self.tst_handlers_group = [list() for i in range(len(tst_datasets_group))]
        for data_name in all_datasets:
            trn_flag = data_name in trn_datasets
            tst_flag = data_name in all_tst_datasets
            handler = DataHandler(data_name)
            if trn_flag:
                self.trn_handlers.append(handler)
            if tst_flag:
                for i in range(len(tst_datasets_group)):
                    if data_name in tst_datasets_group[i]:
                        self.tst_handlers_group[i].append(handler)
        self.make_joint_trn_loader() # 3.3: cross domain model training,

    # Create a joint DataLoader that mixes batches from all training datasets.
    # Each dataset is split into batches of size `args.batch`.
    # These batches are combined into a unified dataset (JointTrnData),
    # where each sample index corresponds to a *full batch* from one original dataset.
    # This allows multi-domain training: each iteration samples a batch from
    # one dataset, while the overall DataLoader shuffles across all datasets' batches.
    # (Note: DataLoader uses batch_size=1 because JointTrnData already returns a full batch.)
    def make_joint_trn_loader(self):
        loader_datasets = []
        for trn_handler in self.trn_handlers:
            tem_dataset = trn_handler.trn_loader.dataset
            loader_datasets.append(tem_dataset)
        joint_dataset = JointTrnData(loader_datasets)
        self.joint_trn_loader = data.DataLoader(joint_dataset, batch_size=1, shuffle=True, num_workers=4,pin_memory=True)
    
    def remake_initial_projections(self):
        for i in range(len(self.trn_handlers)):
            trn_handler = self.trn_handlers[i]
            trn_handler.make_projectors()

class DataHandler:
    def __init__(self, data_name):
        self.data_name = data_name
        self.get_data_files()
        log(f'Loading dataset {data_name}')
        self.topo_encoder = TopoEncoder()
        self.load_data()
    
    def get_data_files(self):
        predir = f'zero-shot datasets/{self.data_name}/'
        if os.path.exists(predir + 'feats.pkl'):
            self.feat_file = predir + 'feats.pkl'
        else:
            self.feat_file = None
        self.trnfile = predir + 'trn_mat.pkl'
        self.tstfile = predir + 'tst_mat.pkl'
        self.fewshotfile = predir + 'partial_mat_{shot}.pkl'.format(shot=args.ratio_fewshot)
        self.valfile = predir + 'val_mat.pkl'

    def load_one_file(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret
    
    def load_feats(self, filename):
        try:
            with open(filename, 'rb') as fs:
                feats = pickle.load(fs)
        except Exception as e:
            print(filename + str(e))
            exit()
        return feats

    def normalize_adj(self, mat, log=False):
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        if mat.shape[0] == mat.shape[1]:
            return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
        else:
            tem = d_inv_sqrt_mat.dot(mat)
            col_degree = np.array(mat.sum(axis=0))
            d_inv_sqrt = np.reshape(np.power(col_degree, -0.5), [-1])
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
            return tem.dot(d_inv_sqrt_mat).tocoo()
    
    def unique_numpy(self, row, col):
        hash_vals = row * args.node_num + col
        hash_vals = np.unique(hash_vals).astype(np.int64)
        col = hash_vals % args.node_num
        row = (hash_vals - col).astype(np.int64) // args.node_num
        return row, col

    def make_torch_adj(self,
                       mat: sp.coo_matrix,
                       *,
                       unidirectional_for_asym: bool = False,
                       keep_weight: bool = True,
                       symmetrize: bool = False):
        """
        若 mat 不是方阵而且本次需要给 GNN（TopoEncoder / Light-GCN）使用，
        先扩展成 UI 方阵，再做后续处理。
        """
        # ---------- (0) 把 item 端 id 复原 ----------
        if mat.shape[0] != mat.shape[1]:  # 二分图
            user_num, item_num = mat.shape
            if mat.col.min() >= user_num:  # item 整体平移过
                mat.col -= user_num  # 只改 col！
        else:  # 方阵
            if mat.row.max() >= mat.shape[0]:
                raise ValueError(f'{self.data_name}: node id 超界')

        # ---------- (1) 如有必要，先拼 UI 方阵 ----------
        if mat.shape[0] != mat.shape[1] and not unidirectional_for_asym:
            user_num, item_num = mat.shape
            zeroUU = sp.csr_matrix((user_num, user_num), dtype=mat.dtype)
            zeroII = sp.csr_matrix((item_num, item_num), dtype=mat.dtype)
            mat_ui = sp.vstack([sp.hstack([zeroUU, mat]),
                                sp.hstack([mat.transpose(), zeroII])]).tocoo()
            # 再继续用下方“方阵”代码处理
            mat = mat_ui

        # ---------- (2) 若要求对称化 ----------
        if symmetrize:
            row = np.concatenate([mat.row, mat.col])
            col = np.concatenate([mat.col, mat.row])
            dat = np.concatenate([mat.data, mat.data]) if keep_weight \
                else np.ones(2 * mat.nnz, dtype=np.float32)
        else:
            row, col = mat.row, mat.col
            dat = mat.data if keep_weight else np.ones_like(mat.data)

        # 去重
        uniq, idx = np.unique(row * mat.shape[1] + col, return_index=True)
        row, col, dat = row[idx], col[idx], dat[idx]

        # ---------- (3) 可选归一化（只有对称化 + 无向 LightGCN 时才做） ----------
        if symmetrize and (not unidirectional_for_asym):
            deg = np.bincount(row, weights=np.abs(dat),
                              minlength=mat.shape[0]).astype(np.float32)
            deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
            dat = dat * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # ---------- (4) 转 torch sparse ----------
        idxs = t.tensor([row, col], dtype=t.long)
        vals = t.tensor(dat, dtype=t.float32)
        shape = t.Size(mat.shape)
        return t.sparse_coo_tensor(idxs, vals, shape)

    def load_data(self):
        tst_mat = self.load_one_file(self.tstfile)
        val_mat = self.load_one_file(self.valfile)
        trn_mat = self.load_one_file(self.trnfile)
        fewshot_mat = self.load_one_file(self.fewshotfile)
        if self.feat_file is not None:
            self.feats = t.from_numpy(self.load_feats(self.feat_file)).float()
            self.feats = self.feats
            args.featdim = self.feats.shape[1]
        else:
            self.feats = None
            args.featdim = args.latdim

        if trn_mat.shape[0] != trn_mat.shape[1]:
            args.user_num, args.item_num = trn_mat.shape
            args.node_num = args.user_num + args.item_num
            print('Dataset: {data_name}, User num: {user_num}, Item num: {item_num}, Node num: {node_num}, Edge num: {edge_num}'.format(data_name=self.data_name, user_num=args.user_num, item_num=args.item_num, node_num=args.node_num, edge_num=trn_mat.nnz))
        else:
            args.node_num = trn_mat.shape[0]
            print('Dataset: {data_name}, Node num: {node_num}, Edge num: {edge_num}'.format(data_name=self.data_name, node_num=args.node_num, edge_num=trn_mat.nnz+val_mat.nnz+tst_mat.nnz))
        if args.tst_mode == 'tst':
            tst_data = TstData(tst_mat, trn_mat)
            self.tst_loader = data.DataLoader(tst_data, batch_size=args.tst_batch, shuffle=False, num_workers=4,pin_memory=True)
            self.tst_input_adj = self.make_torch_adj(trn_mat)
        elif args.tst_mode == 'val':
            tst_data = TstData(val_mat, trn_mat)
            self.tst_loader = data.DataLoader(tst_data, batch_size=args.tst_batch, shuffle=False, num_workers=4,pin_memory=True)
            self.tst_input_adj = self.make_torch_adj(fewshot_mat)
        else:
            raise Exception('Specify proper test mode')

        if args.trn_mode == 'fewshot':
            self.trn_mat = fewshot_mat
            trn_data = TrnData(self.trn_mat)
            self.trn_loader = data.DataLoader(trn_data, batch_size=args.batch, shuffle=True, num_workers=4,pin_memory=True)
            self.trn_input_adj = self.make_torch_adj(fewshot_mat)
            if args.tst_mode == 'val':
                self.trn_input_adj = self.tst_input_adj
            else:
                self.trn_input_adj = self.make_torch_adj(fewshot_mat)
        elif args.trn_mode == 'train-all':
            self.trn_mat = trn_mat
            trn_data = TrnData(self.trn_mat)
            self.trn_loader = data.DataLoader(trn_data, batch_size=args.batch, shuffle=True, num_workers=4,pin_memory=True)
            if args.tst_mode == 'tst':
                self.trn_input_adj = self.tst_input_adj
            else:
                self.trn_input_adj = self.make_torch_adj(trn_mat)
        else:
            raise Exception('Specify proper train mode')   

        if self.trn_mat.shape[0] == self.trn_mat.shape[1]:
            self.asym_adj = self.trn_input_adj
        else:
            self.asym_adj = self.make_torch_adj(self.trn_mat, unidirectional_for_asym=True)
        self.make_projectors()
        self.reproj_steps = max(len(self.trn_loader.dataset) // (10 * args.batch), args.proj_trn_steps)
        self.ratio_500_all = 500 / len(self.trn_loader)

        # expose edge_index / edge_weight to MSGNN expert
        coo_adj = self.asym_adj.coalesce()
        self.edge_index = coo_adj.indices().to(t.long)  # [2, E]
        self.edge_weight = coo_adj.values().to(t.float)  # [E]
    
    def make_projectors(self):
        with t.no_grad():
            projectors = []
            if args.proj_method == 'adj_svd' or args.proj_method == 'both':
                tem = self.asym_adj.to(args.devices[0])
                projectors = [Adj_Projector(tem)]
            if self.feats is not None and args.proj_method != 'adj_svd':
                tem = self.feats.to(args.devices[0])
                projectors.append(Feat_Projector(tem))
            assert args.tst_mode == 'tst' and args.trn_mode == 'train-all' or args.tst_mode == 'val' and args.trn_mode == 'fewshot'
            feats = projectors[0]()
            if len(projectors) == 2:
                feats2 = projectors[1]()
                feats = feats + feats2

            try:
                self.projectors = self.topo_encoder(self.trn_input_adj.to(args.devices[0]), feats.to(args.devices[0])).detach().cpu()
            except Exception:
                print(f'{self.data_name} memory overflow')
                mean, std = feats.mean(dim=-1, keepdim=True), feats.std(dim=-1, keepdim=True)
                tem_adj = self.trn_input_adj.to(args.devices[0])
                mem_cache = 256
                projectors_list = []
                for i in range(feats.shape[1] // mem_cache):
                    st, ed = i * mem_cache, (i + 1) * mem_cache
                    tem_feats = (feats[:, st:ed] - mean) / (std + 1e-8)
                    tem_feats = self.topo_encoder(tem_adj, tem_feats.to(args.devices[0]), normed=True).detach().cpu()
                    projectors_list.append(tem_feats)
                self.projectors = t.concat(projectors_list, dim=-1)
            t.cuda.empty_cache()

class TstData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0
        tstLocs = [None] * coomat.shape[0]
        tst_nodes = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tst_nodes.add(row)
        tst_nodes = np.array(list(tst_nodes))
        self.tst_nodes = tst_nodes
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tst_nodes)

    def __getitem__(self, idx):
        return self.tst_nodes[idx]

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.ancs, self.poss = coomat.row, coomat.col
        self.negs = np.zeros(len(self.ancs)).astype(np.int32)
        self.cand_num = coomat.shape[1]
        self.neg_shift = 0 if coomat.shape[0] == coomat.shape[1] else coomat.shape[0]
        self.poss = coomat.col + self.neg_shift
        self.neg_sampling()
    
    def neg_sampling(self):
        self.negs = np.random.randint(self.cand_num + self.neg_shift, size=self.poss.shape[0])

    def __len__(self):
        return len(self.ancs)
    
    def __getitem__(self, idx):
        return self.ancs[idx], self.poss[idx] , self.negs[idx]

class JointTrnData(data.Dataset):
    def __init__(self, dataset_list):
        self.batch_dataset_ids = []
        self.batch_st_ed_list = []
        self.dataset_list = dataset_list
        for dataset_id, dataset in enumerate(dataset_list):
            samp_num = len(dataset) // args.batch + (1 if len(dataset) % args.batch != 0 else 0)
            for j in range(samp_num):
                self.batch_dataset_ids.append(dataset_id)
                st = j * args.batch
                ed = min((j + 1) * args.batch, len(dataset))
                self.batch_st_ed_list.append((st, ed))
    
    def neg_sampling(self):
        for dataset in self.dataset_list:
            dataset.neg_sampling()

    def __len__(self):
        return len(self.batch_dataset_ids)
    
    def __getitem__(self, idx):
        st, ed = self.batch_st_ed_list[idx]
        dataset_id = self.batch_dataset_ids[idx]
        return *self.dataset_list[dataset_id][st: ed], dataset_id