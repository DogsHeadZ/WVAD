import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

    def forward(self, ref_rgb_feat, ref_flow_feat, sup_rgb_feat, sup_flow_feat):
        # ref_rgb_feat: (B, N1, F), ref_flow_feat: (B, N1, F), sup_rgb_feat: (B, N2, F), sup_flow_feat: (B, N2, F)

        F = ref_rgb_feat.shape[-1]
        # 视频内部挑选较难区分的样本，即平均相似性最小的K个样本
        intra_sim = torch.matmul(ref_flow_feat, ref_flow_feat.permute(0, 2, 1)) # [B, N1, N1]
        intra_sim = torch.mean(intra_sim, dim=2)  # [B, N1]
        intra_topK = 3
        k_idx = torch.topk(intra_sim, intra_topK, dim=1, largest=False)[1]  # [B, K1]
        k_idx = k_idx.unsqueeze(2).expand([-1, -1, F])
        k_ref_rgb_feat = torch.gather(ref_rgb_feat, 1, k_idx)   #[B, K1, F]
        k_ref_flow_feat = torch.gather(ref_flow_feat, 1, k_idx)

        # 视频间fusion
        inter_sim = torch.matmul(sup_flow_feat, k_ref_flow_feat.permute(0, 2, 1))  # [B, N2, K1]
        inter_sim_mean = torch.mean(inter_sim, dim=2)  # [B, N2]
        inter_topK = 2
        # 调出sup中相似性最高的K个样本
        k_idx = torch.topk(inter_sim_mean, inter_topK, dim=1)[1]  # [B, K2]
        k_idx = k_idx.unsqueeze(2).expand([-1, -1, F])
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