import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

    def forward(self, ref_rgb_feat, ref_flow_feat, sup_rgb_feat, sup_flow_feat):
        # ref_rgb_feat: (B, N1, F), ref_flow_feat: (B, N1, F), sup_rgb_feat: (B, N2, F), sup_flow_feat: (B, N2, F)

        F_len = ref_rgb_feat.shape[-1]
        # 视频内部挑选较难区分的样本，即平均相似性最小的K个样本
        intra_sim = torch.matmul(ref_flow_feat/(torch.norm(ref_flow_feat, p=2, dim=2).unsqueeze(2)),
                                 (ref_flow_feat/(torch.norm(ref_flow_feat, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1)) # [B, N1, N1]
        intra_sim = F.softmax(intra_sim, dim=2)
        intra_sim = torch.mean(intra_sim, dim=2)  # [B, N1]
        intra_topK = 8
        k_idx = torch.topk(intra_sim, intra_topK, dim=1, largest=False)[1]  # [B, K1]
        k_idx = k_idx.unsqueeze(2).expand([-1, -1, F_len])
        k_ref_rgb_feat = torch.gather(ref_rgb_feat, 1, k_idx)   #[B, K1, F]
        k_ref_flow_feat = torch.gather(ref_flow_feat, 1, k_idx)

        # 视频间fusion
        inter_sim = torch.matmul(sup_flow_feat/(torch.norm(sup_flow_feat, p=2, dim=2).unsqueeze(2)),
                                 (ref_flow_feat/(torch.norm(ref_flow_feat, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1))  # [B, N2, K1]
        inter_sim = F.softmax(inter_sim, dim=2)
        inter_sim_mean = torch.mean(inter_sim, dim=2)  # [B, N2]
        inter_topK = 8
        # 调出sup中相似性最高的K个样本
        k_idx = torch.topk(inter_sim_mean, inter_topK, dim=1)[1]  # [B, K2]
        k_idx = k_idx.unsqueeze(2).expand([-1, -1, F_len])
        k_sup_rgb_feat = torch.gather(sup_rgb_feat, 1, k_idx)   #[B, K1, F]

        fusion_rgb_feat = torch.cat([k_ref_rgb_feat, k_sup_rgb_feat], dim=1) # 在视频内部维度上进行拼接  [B, K1+K2, F]
        # fusion_rgb_feat = k_ref_rgb_feat + k_sup_rgb_feat

        return k_ref_rgb_feat, k_sup_rgb_feat

def make_fc(dim_in, hidden_dim):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''

    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc

class Relation(nn.Module):
    def __init__(self, feat_dim):
        super(Relation, self).__init__()
        fcs, Wqs, Wks, Wvs = [], [], [], []

        layernum = 2
        input_size = feat_dim
        representation_size = 2048

        self.embed_dim = 64
        self.groups = 16
        self.feat_dim = representation_size

        for i in range(layernum + 1):
            r_size = input_size if i == 0 else representation_size

            if i != layernum:
                fcs.append(make_fc(r_size, representation_size))
            Wqs.append(make_fc(self.feat_dim, self.feat_dim))
            Wks.append(make_fc(self.feat_dim, self.feat_dim))
            Wvs.append(nn.Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0,
                              groups=self.groups))
            torch.nn.init.normal_(Wvs[i].weight, std=0.01)
            torch.nn.init.constant_(Wvs[i].bias, 0)

        self.fcs = nn.ModuleList(fcs)
        self.Wqs = nn.ModuleList(Wqs)
        self.Wks = nn.ModuleList(Wks)
        self.Wvs = nn.ModuleList(Wvs)

    def attention_module_multi_head(self, roi_feat, ref_feat, feat_dim=2048, dim=(2048, 2048, 2048), group=16, index=0):
        """

        :param roi_feat: [num_rois, feat_dim]
        :param ref_feat: [num_nongt_rois, feat_dim]
        :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # multi head
        assert dim[0] == dim[1]

        q_data = self.Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = self.Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff, [group, num_rois, num_nongt_rois]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        weighted_aff = aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)

        return output

class VideoRelation(nn.Module):
    def __init__(self, feat_dim):
        super(VideoRelation, self).__init__()
        fcs, Wqs, Wks, Wvs = [], [], [], []

        layernum = 1
        input_size = feat_dim
        representation_size = feat_dim

        self.embed_dim = 64
        self.groups = 16
        self.feat_dim = representation_size

        for i in range(layernum + 1):
            r_size = input_size if i == 0 else representation_size

            if i != layernum:
                fcs.append(make_fc(r_size, representation_size))
            Wqs.append(make_fc(self.feat_dim, self.feat_dim))
            Wks.append(make_fc(self.feat_dim, self.feat_dim))
            Wvs.append(nn.Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0,
                              groups=self.groups))
            torch.nn.init.normal_(Wvs[i].weight, std=0.01)
            torch.nn.init.constant_(Wvs[i].bias, 0)

        self.fcs = nn.ModuleList(fcs)
        self.Wqs = nn.ModuleList(Wqs)
        self.Wks = nn.ModuleList(Wks)
        self.Wvs = nn.ModuleList(Wvs)

    def attention_module_multi_head(self, roi_feat, ref_feat, feat_dim=2048, dim=(2048, 2048, 2048), group=16, index=0):
        """

        :param roi_feat: [bs, N1, feat_dim]
        :param ref_feat: [bs, N2, feat_dim]
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        bs = roi_feat.shape[0]
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # multi head
        assert dim[0] == dim[1]

        q_data = self.Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(q_data.shape[0], -1, group, int(dim_group[0]))
        # q_data_batch, [bs, group, N1, dim_group[0]]
        q_data_batch = q_data_batch.permute(0, 2, 1, 3)

        k_data = self.Wks[index](ref_feat)
        k_data_batch = k_data.reshape(k_data.shape[0], -1, group, int(dim_group[1]))
        # k_data_batch, [bs, group, N2, dim_group[1]]
        k_data_batch = k_data_batch.permute(0, 2, 1, 3)

        # v_data, [bs, N2, feat_dim]
        v_data = ref_feat

        # aff, [bs, group, N1, N2]
        aff = torch.matmul(q_data_batch, k_data_batch.permute(0,1, 3, 2))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [ns, N1, group, N2]
        aff_scale = aff_scale.permute(0, 2, 1, 3)

        weighted_aff = aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=3)

        aff_softmax_reshape = aff_softmax.reshape(-1, aff_softmax.shape[1] * aff_softmax.shape[2], aff_softmax.shape[3])

        # output_t, [bs, N1 * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [bs, N1, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)
        output = output.reshape(bs, -1, output.shape[-1])

        return output


class SA(nn.Module):
    def __init__(self, feat_dim):
        super(SA, self).__init__()
        self.relation = VideoRelation(feat_dim)


    def forward(self, ref_feat, sup_feat):
        # feat: [B, N, F]

        out = self.relation.attention_module_multi_head(ref_feat, sup_feat)

        return out




class MotionMemory(nn.Module):
    def __init__(self):
        super(MotionMemory, self).__init__()
        self.m_items = F.normalize(torch.rand((10, 2048), dtype=torch.float),
                              dim=1)  # Initialize the memory items


    def read(self, feat):
        # feat: [B, N, F], mitems: [M, F]
        mitems = self.m_items
        bs, N, F_len = feat.size()
        feat = feat.contiguous().view(-1, F_len)

        sim = torch.matmul(F.normalize(feat, dim=1), F.normalize(mitems, dim=1).permute(1, 0))  # [B*N, M]
        sim = F.softmax(sim, dim=1)

        mitems_feat = torch.matmul(sim, mitems)  # [B*N, F]
        feat_update = feat + mitems_feat
        feat_update = feat_update.view(bs, N, F_len)
        return feat_update


    def update(self, feat):
        mitems = self.m_items
        M, F_len = mitems.size()
        feat = feat.contiguous().view(-1, F_len)
        sim = torch.matmul(F.normalize(feat, dim=1), F.normalize(mitems, dim=1).permute(1, 0))  # [B*N, M]
        sim = F.softmax(sim, dim=1)
        k_idx = torch.topk(sim, 1, dim=1)[1].squeeze(1)  #[B*N, ]
        sim_t = F.softmax(sim, dim=0)

        mitems_update = torch.zeros_like(mitems).cuda()
        for i in range(M):
            idx = torch.nonzero(k_idx == i)
            if idx.shape[0] != 0:
                mitems_update[i] = torch.sum((sim_t[idx, i] / torch.max(sim_t[:, i])) * feat[idx].squeeze(1), dim=0)
            else:
                mitems_update[i] = 0
        mitems_update = F.normalize(mitems_update + mitems, dim=1)

        return mitems_update


    def forward(self, feat, train=True):
        # feat: [B, N, F]
        feat_update = self.read(feat)

        if not train:
            return feat_update

        mem_update = self.update(feat)
        self.m_items = mem_update
        return feat_update



if __name__ == '__main__':
    import numpy as np


    memory = MotionMemory()
    input = torch.rand([32,8,2048], dtype=torch.float)
    result = memory(input)
    print(result.shape)