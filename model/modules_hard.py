import torch
import torch.nn as nn
from torch.nn import functional as F
import math
class Hard_score_sample_generator(nn.Module):
    def __init__(self):
        super(Hard_score_sample_generator, self).__init__()

    def forward(self, feat, scores_rgb, scores_flow):
        F_len = feat.shape[-1]
        hard_topK = 10  # 挑选出abnormal video中的x个困难样本，即异常分数值接近于0.5
        hard_topK_idx = torch.topk(torch.abs_(scores_rgb - 0.5), hard_topK, dim=1, largest=False)[1]
        hard_topK_flow_scores = torch.gather(scores_flow, 1, hard_topK_idx)
        hard_topK_feat = torch.gather(feat, 1, hard_topK_idx.unsqueeze(2).expand([-1, -1, F_len]))

        hard_topK_abn = 3  # 从x个困难样本中挑选出x个flow score为异常的样本
        hard_topK_idx_abn = torch.topk(hard_topK_flow_scores, hard_topK_abn, dim=1)[1]
        hard_topK_feat_abn = torch.gather(hard_topK_feat, 1, hard_topK_idx_abn.unsqueeze(2).expand([-1, -1, F_len]))

        hard_topK_nor = 1  # 从x个困难样本中挑选出x个flow score为正常的样本
        hard_topK_idx_nor = torch.topk(hard_topK_flow_scores, hard_topK_nor, dim=1, largest=False)[1]
        hard_topK_feat_nor = torch.gather(hard_topK_feat, 1, hard_topK_idx_nor.unsqueeze(2).expand([-1, -1, F_len]))

        conf_nor_topK = 2  # 将abnormal video中置信度高的x个clip视为normal
        conf_k_idx_nor = torch.topk(scores_rgb, conf_nor_topK, dim=1, largest=False)[1]
        conf_k_idx_nor = conf_k_idx_nor.unsqueeze(2).expand([-1, -1, F_len])
        conf_rgb_feat_nor = torch.gather(feat, 1, conf_k_idx_nor)

        conf_abn_topK = 10  # 挑选出abnormal video中置信度高的x个clip作为abnormal
        conf_k_idx_abn = torch.topk(scores_rgb, conf_abn_topK, dim=1, largest=True)[1]
        conf_k_idx_abn = conf_k_idx_abn.unsqueeze(2).expand([-1, -1, F_len])
        conf_rgb_feat_abn = torch.gather(feat, 1, conf_k_idx_abn)

        return hard_topK_feat_nor, hard_topK_feat_abn, conf_rgb_feat_nor, conf_rgb_feat_abn

class Hard_sim_sample_generator(nn.Module):
    def __init__(self):
        super(Hard_sim_sample_generator, self).__init__()

    def forward(self, feat):
        # ref_rgb_feat: (B, N, F)
        F_len = feat.shape[-1]
        # 视频内部挑选较难区分的样本，即平均相似性最小的K个样本
        intra_sim = torch.matmul(feat / (torch.norm(feat, p=2, dim=2).unsqueeze(2)),
                                 (feat / (torch.norm(feat, p=2, dim=2).unsqueeze(2))).permute(0, 2, 1))  # [B, N, N]
        intra_sim = F.softmax(intra_sim, dim=2)
        intra_sim = torch.mean(intra_sim, dim=2)  # [B, N]

        hard_sim_topK = 8  # 挑出相似性小的x个样本
        hard_k_idx = torch.topk(intra_sim, hard_sim_topK, dim=1, largest=False)[1]  # [B, K1]
        hard_k_idx = hard_k_idx.unsqueeze(2).expand([-1, -1, F_len])
        hard_k_feat = torch.gather(feat, 1, hard_k_idx)  # [B, K1, F]

        conf_sim_topK = 8  # 挑出相似性大的x个样本
        conf_k_idx = torch.topk(intra_sim, conf_sim_topK, dim=1, largest=True)[1]  # [B, K2]
        conf_k_idx = conf_k_idx.unsqueeze(2).expand([-1, -1, F_len])
        conf_k_feat = torch.gather(feat, 1, conf_k_idx)  # [B, K2, F]

        return hard_k_feat, conf_k_feat


def make_fc(dim_in, hidden_dim):

    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


class VideoRelation(nn.Module):
    def __init__(self, feat_dim):
        super(VideoRelation, self).__init__()
        fcs, Wqs, Wks, Wvs = [], [], [], []

        self.layernum = 3
        input_size = feat_dim
        representation_size = feat_dim

        self.embed_dim = 64
        self.groups = 16
        self.feat_dim = representation_size

        for i in range(self.layernum):
            r_size = input_size if i == 0 else representation_size

            if i != self.layernum:
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
        # linear_out, [bs, N1, dim[2], 1, 1]
        linear_out = self.Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)
        output = output.reshape(bs, -1, output.shape[-1])

        return output

    def forward(self, ref_feat, sup_feat, index=0):
        # feat: [B, N, F]
        # ref_feat = F.relu(self.fcs[index](ref_feat))
        # sup_feat = F.relu(self.fcs[0](sup_feat))
        #
        # attention = self.attention_module_multi_head(ref_feat, sup_feat, index=index)
        # ref_feat = ref_feat + attention

        sup_feat = F.relu(self.fcs[0](sup_feat))
        for i in range(1):
            ref_feat = F.relu(self.fcs[i](ref_feat))
            attention = self.attention_module_multi_head(ref_feat, sup_feat, index=i)
            ref_feat = ref_feat + attention

        return ref_feat


# https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.4.1_to_1.1.0/lib/non_local_gaussian.py
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class Aggregate(nn.Module):
    def __init__(self, feat_dim):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = feat_dim
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=feat_dim, out_channels=512, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=feat_dim, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=feat_dim, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)


    def forward(self, x):
            # x: (B, T, F)
            out = x.permute(0, 2, 1)    # 做了个转置是为了在时间维度上应用attention
            residual = out

            out1 = self.conv_1(out)
            out2 = self.conv_2(out)

            out3 = self.conv_3(out)
            out_d = torch.cat((out1, out2, out3), dim = 1)
            out = self.conv_4(out)
            out = self.non_local(out)
            out = torch.cat((out_d, out), dim=1)
            out = self.conv_5(out)   # fuse all the features together
            out = out + residual
            out = out.permute(0, 2, 1)
            # out: (B, T, F)

            return out

if __name__ == '__main__':
    input = torch.rand([32,8,2048], dtype=torch.float)
