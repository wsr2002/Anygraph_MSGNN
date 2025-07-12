import torch as t
import torch.nn as nn
from MSGNN import MSGNN_link_prediction        # 已复制
from torch_geometric_signed_directed.utils import in_out_degree
from params import args

class MSGNNExpert(nn.Module):
    """
    Exchange AnyGraph's Expert to MSGNN。
    forward() input:
        1) feats (E1 in AnyGraph)       : [N, F]
        2) batch_edges (= (ancs, poss))   : positive edge (negative ones are sampled in cal_loss)
    """
    def __init__(self, args, num_input_feat):
        super().__init__()
        self.args      = args
        self.num_feat  = num_input_feat        # 2/4/…，只是记录
        self.in_dim    = None                  # 真正建 GNN 时再写
        self.gnn       = None
        self.optimizer = None
        self.loss_fn   = nn.CrossEntropyLoss()

        # 图信息（bind_dataset 之后才有）
        self.edge_index  = None
        self.edge_weight = None

    def _pad_feat_if_needed(self, feats):
        needed_rows = int(self.edge_index.max()) + 1  # max node id used by edges
        cur_rows = feats.size(0)
        if needed_rows > cur_rows:  # 还差多少行
            pad = feats.new_zeros(needed_rows - cur_rows, feats.size(1))
            feats = t.cat([feats, pad], dim=0)
        return feats

    def _compact_graph(self, edge_index, feats):
        """
        把 edge_index 的节点 id 压缩到 [0, n') 区间，
        同时返回重新排列后的 feats 以及 id 映射后的 edge_index
        """
        device = edge_index.device
        uniq = t.unique(edge_index)
        # 建映射 old_id -> new_id
        new_ids = t.arange(uniq.size(0), device=device)
        id_map = t.full((uniq.max() + 1,), -1, device=device, dtype=t.long)
        id_map[uniq] = new_ids
        edge_index = id_map[edge_index]  # 映射
        feats = feats[uniq]  # 同步裁剪 / 重排
        return edge_index, feats
    # ---------- 延迟构建 / 重建 GNN ----------
    def _build_gnn(self, feat_dim: int):
        """若第一次调用或 feat_dim 改变，则重建 GNN 和 Optimizer"""
        self.in_dim = feat_dim
        self.gnn = MSGNN_link_prediction(
            q=args.q, K=args.K,
            num_features=feat_dim,
            hidden=args.hidden,
            label_dim=2,
            trainable_q=args.trainable_q,
            layer=args.num_layers,
            dropout=args.dropout,
            normalization=args.normalization,
            cached=not args.trainable_q
        ).to(args.devices[0])

        # 重新创建优化器
        self.optimizer = t.optim.Adam(
            self.gnn.parameters(), lr=args.lr, weight_decay=args.reg
        )
        t.cuda.empty_cache()

    # ---------- 把数据集的稀疏邻接矩阵缓存进来 ----------
    def bind_dataset(self, handler):
        adj = handler.trn_input_adj.coalesce()
        edge_index  = adj.indices().to(args.devices[0])
        self.edge_weight = adj.values().to(args.devices[0])
        # -------- NEW: 压缩 id，使 edge_index.max < feats.size(0) ----------
        feats = handler.projectors.to(args.devices[0])  # 原 E1
        edge_index, feats = self._compact_graph(edge_index, feats)
        handler.compact_projectors = feats  # 把压缩后的 E1 存回 handler
        # -------------------------------------------------------------------

        self.edge_index = edge_index



    def cal_loss(self, edges, feats):
        if self.gnn is None or feats.size(1) != self.in_dim:
            self._build_gnn(feats.size(1))
        ancs, poss, negs = edges
        device = next(self.gnn.parameters()).device

        # construct positive/negative inquiry edge index
        pos_q = t.stack([ancs, poss], dim=1)  # [B, 2]
        neg_q = t.stack([ancs, negs], dim=1)  # [B, 2]
        q_edges = t.cat([pos_q, neg_q], dim=0)    # [2B, 2]
        labels = t.cat([t.zeros(pos_q.size(0)),
                        t.ones(neg_q.size(0))]).long().to(device)
        # MSGNN forward
        X_real = self._pad_feat_if_needed(feats.to(device))
        X_img = X_real.clone()
        logits = self.gnn(
            X_real, X_img,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            query_edges=q_edges
        )          # [2B, 2]

        loss = self.loss_fn(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with t.no_grad():
            pred = logits.argmax(dim=1).cpu()
            preloss = self.loss_fn(logits, labels).item()
            posloss = (pred[:pos_q.shape[1]]!=0).float().mean().item()
            negloss = (pred[pos_q.shape[1]:]!=1).float().mean().item()
            regloss = 0.0
        loss_dict = dict(preloss=preloss, posloss=posloss,
                         negloss=negloss, regloss=regloss)
        return loss, loss_dict

    # test
    def _encode(self, feats: t.Tensor) -> t.Tensor:
        """
        Return node embeddings of shape [N, 2*hidden].
        No classification head, no query_edges involved.
        """
        device = next(self.gnn.parameters()).device
        real = feats.to(device).float()
        imag = real.clone()

        # 把 Cheb-Conv 串起来
        for cheb in self.gnn.Chebs:
            real, imag = cheb(real, imag,
                              edge_index=self.edge_index,
                              edge_weight=self.edge_weight)
            if self.gnn.activation:
                real, imag = self.gnn.complex_relu(real, imag)

        z = t.cat([real, imag], dim=-1)         # [N , 2*hidden]
        return z

    def pred_for_test(self,
                      batch_data,          # (usrs , trn_masks)
                      cand_size: int,
                      feats: t.Tensor,
                      rerun_embed: bool = True):
        """
        Return score matrix [|usrs| , cand_size] – identical to old AnyGraph.
        """
        usrs, trn_mask = map(lambda x: x.to(args.devices[1]), batch_data)

        if rerun_embed:
            self.final_embeds = self._encode(feats.to(args.devices[1]))
        final_embeds = self.final_embeds  # [N, 2*hidden]

        anc_embeds = final_embeds[usrs]  # [B, 2*hidden]
        cand_embeds = final_embeds[-cand_size:]  # [cand_size, 2*hidden]

        # ------- 关键修改：确保索引和值在同一块 GPU 上 -----------------
        values = t.ones(trn_mask.shape[1], device=trn_mask.device)  # ★
        mask_mat = t.sparse.FloatTensor(
            trn_mask,  # indices (2 , nnz)
            values,  # values  (nnz)
            t.Size([usrs.shape[0], cand_size])
        )
        # ----------------------------------------------------------------
        dense_mask = mask_mat.to_dense()
        all_preds = anc_embeds @ cand_embeds.T * (1 - dense_mask) - dense_mask * 1e8
        return all_preds

    def attempt(self, topo_embeds, dataset):
        final_embeds = topo_embeds
        rows, cols, negs = list(map(lambda x: t.from_numpy(x).long().to(args.devices[1]), [dataset.ancs, dataset.poss, dataset.negs]))
        if rows.shape[0] > args.attempt_cache:
            random_perm = t.randperm(rows.shape[0], device=args.devices[0])
            pck_perm = random_perm[:args.attempt_cache]
            rows = rows[pck_perm]
            cols = cols[pck_perm]
            negs = negs[pck_perm]
        while True:
            try:
                row_embeds = final_embeds[rows]
                col_embeds = final_embeds[cols]
                neg_embeds = final_embeds[negs]
                score = ((row_embeds * col_embeds).sum(-1) - (row_embeds * neg_embeds).sum(-1)).sigmoid().mean().item()
                break
            except Exception:
                args.attempt_cache = args.attempt_cache // 2
                random_perm = t.randperm(rows.shape[0], device=args.devices[0])
                pck_perm = random_perm[:args.attempt_cache]
                rows = rows[pck_perm]
                cols = cols[pck_perm]
                negs = negs[pck_perm]
        t.cuda.empty_cache()
        return score
