import torch
import torch.nn as nn
from torch.nn import functional as F

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
        inter_topK = 4
        # 调出sup中相似性最高的K个样本
        k_idx = torch.topk(inter_sim_mean, inter_topK, dim=1)[1]  # [B, K2]
        k_idx = k_idx.unsqueeze(2).expand([-1, -1, F_len])
        k_sup_rgb_feat = torch.gather(sup_rgb_feat, 1, k_idx)   #[B, K1, F]

        fusion_rgb_feat = torch.cat([k_ref_rgb_feat, k_sup_rgb_feat], dim=1) # 在视频内部维度上进行拼接  [B, K1+K2, F]

        return fusion_rgb_feat


class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()


    def forward(self, feat):
        # feat: [B, N, F]


        return feat


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()


    def forward(self, feat):
        # feat: [B, N, F]


        return feat

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