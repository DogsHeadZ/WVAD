import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from model.modules import Fusion, MotionMemory, SA


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


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
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=3,
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

class CoModel(nn.Module):
    def __init__(self, n_features, batch_size):
        super(CoModel, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.Aggregate = Aggregate(len_feature=2048)
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

        self.fusion = Fusion()
        self.memory = MotionMemory()
        self.SA = SA()
        self.classifier = nn.Sequential(nn.Dropout(0.8),nn.Linear(2048,1))

        self.Softmax = nn.Softmax(dim=-1)


    def forward(self, ref_rgb, ref_flow, normal_rgb, normal_flow, abnormal_rgb, abnormal_flow, mode='normal', tencrop=True):

        inputs = torch.cat((ref_rgb, normal_rgb, abnormal_rgb), 0)
        bs, ncrops, T, F = inputs.size()

        inputs = inputs.view(-1, T, F)
        inputs = self.Aggregate(inputs)
        inputs = self.drop_out(inputs)
        features = inputs     # （bs*ncrops, T, F)
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1)   # 对这个帧的10个crops取平均得到这帧的分数[bs, T]

        bs = bs // 3
        ref_p_scores = scores[0:bs]
        nor_scores = scores[bs: 2*bs]
        abn_scores = scores[2*bs:]

        if ref_p_scores.shape[0] == 1:  # this is for inference
            nor_scores = ref_p_scores
            abn_scores = ref_p_scores
            return ref_p_scores.mean(1)

        if tencrop:
            # ref_rgb, ref_flow, normal_rgb, normal_flow, abnormal_rgb, abnormal_flow = \
            #     ref_rgb.mean(1), ref_flow.mean(1), normal_rgb.mean(1), normal_flow.mean(1), abnormal_rgb.mean(1), abnormal_flow.mean(1)
            ref_p_scores, abn_scores = ref_p_scores.view(bs*ncrops, -1), abn_scores.view(bs*ncrops, -1)

            ref_rgb, ref_flow, normal_rgb, normal_flow, abnormal_rgb, abnormal_flow = \
                ref_rgb.view(bs*ncrops, T, -1), ref_flow.view(bs*ncrops, T, -1), normal_rgb.view(bs*ncrops, T, -1), \
                    normal_flow.view(bs*ncrops, T, -1), abnormal_rgb.view(bs*ncrops, T, -1), abnormal_flow.view(bs*ncrops, T, -1)

        # if bs == 1:
        #     xxx = self.memory(ref_rgb, train=False)  # todo
        #
        #     ref_scores = self.Softmax(self.classifier(xxx))
        #
        #     if tencrop:
        #         ref_scores = ref_scores.view(bs, ncrops, -1).mean(1)
        #     return ref_scores

        if mode == 'normal':
            # 当ref为normal时
            nor_topK = 2   #将abnormal video中的x个clip视为normal
            abn_k_idx_nor = torch.topk(abn_scores, nor_topK, dim=1, largest=False)[1]
            abn_k_idx_nor = abn_k_idx_nor.unsqueeze(2).expand([-1, -1, F])
            abn_rgb_feat_nor = torch.gather(abnormal_rgb, 1, abn_k_idx_nor)
            abn_flow_feat_nor = torch.gather(abnormal_flow, 1, abn_k_idx_nor)

            abn_topK = 8   #挑选出abnormal video中x个clip作为abnormal
            abn_k_idx_abn = torch.topk(abn_scores, abn_topK, dim=1, largest=True)[1]
            abn_k_idx_abn = abn_k_idx_abn.unsqueeze(2).expand([-1, -1, F])
            abn_rgb_feat_abn = torch.gather(abnormal_rgb, 1, abn_k_idx_abn)
            abn_flow_feat_abn = torch.gather(abnormal_flow, 1, abn_k_idx_abn)

            nor_rgb_feat = torch.cat([normal_rgb, abn_rgb_feat_nor], dim=1)  # 在视频内部维度上进行拼接
            nor_flow_feat = torch.cat([normal_flow, abn_flow_feat_nor], dim=1)

            fusion_feat = self.fusion(ref_rgb, ref_flow, nor_rgb_feat, nor_flow_feat)
            ref_attn_feat = self.SA(fusion_feat)
            sup_attn_feat = self.SA(abn_rgb_feat_abn) # for rank loss

        else:
            # 当ref为abnormal时
            ref_topK = 10
            ref_k_idx_abn = torch.topk(ref_p_scores, ref_topK, dim=1, largest=True)[1]
            ref_k_idx_abn = ref_k_idx_abn.unsqueeze(2).expand([-1, -1, F])
            ref_rgb_feat_abn = torch.gather(ref_rgb, 1, ref_k_idx_abn)
            ref_flow_feat_abn = torch.gather(ref_flow, 1, ref_k_idx_abn)

            abn_topK = 10
            abn_k_idx_abn = torch.topk(abn_scores, abn_topK, dim=1, largest=True)[1]
            abn_k_idx_abn = abn_k_idx_abn.unsqueeze(2).expand([-1, -1, F])
            abn_rgb_feat_abn = torch.gather(abnormal_rgb, 1, abn_k_idx_abn)
            abn_flow_feat_abn = torch.gather(abnormal_flow, 1, abn_k_idx_abn)

            fusion_feat_nor = self.fusion(ref_rgb_feat_abn, ref_flow_feat_abn, abn_rgb_feat_abn, abn_flow_feat_abn)
            ref_attn_feat = self.SA(fusion_feat_nor)
            sup_attn_feat = self.SA(normal_rgb)  # for rank loss

        xxx = self.memory(ref_attn_feat)  # todo

        ref_scores = self.Softmax(self.classifier(xxx))

        if tencrop:
            ref_scores = ref_scores.view(bs, ncrops, -1).mean(1)

        return ref_p_scores, ref_scores

class SimpleModel(nn.Module):
    def __init__(self, n_features, batch_size):
        super(SimpleModel, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.Aggregate = Aggregate(len_feature=2048)
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        bs, ncrops, t, f = out.size()

        out = out.view(-1, t, f)

        # out = self.Aggregate(out)

        # out = self.drop_out(out)

        features = out     # （bs*ncrops, T, F)
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1)   # 对这个帧的10个crops取平均得到这帧的分数
        scores = scores.unsqueeze(dim=2)   # [bs, T, 1]

        normal_scores = scores[0:self.batch_size]
        abnormal_scores = scores[self.batch_size:]


        if normal_scores.shape[0] == 1:  # this is for inference
            abnormal_scores = normal_scores

        score_normal = torch.mean(normal_scores, dim=1)
        score_abnormal = torch.mean(abnormal_scores, dim=1)

        return score_abnormal, score_normal, scores