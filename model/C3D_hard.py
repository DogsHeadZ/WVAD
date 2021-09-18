import torch
from torch import nn
from model.modules_hard import Hard_sim_sample_generator, Hard_score_sample_generator, VideoRelation, Aggregate
from model.I3D_hard import I3D

def load_c3d_pretrained_model(net,checkpoint_path,name=None):
    checkpoint = torch.load(checkpoint_path)
    state_dict = net.state_dict()
    base_dict = {}
    checkpoint_keys = checkpoint.keys()
    if name==None:
        for k, v in state_dict.items():
            for _k in checkpoint_keys:

                if k in _k:
                    base_dict[k] = checkpoint[_k]
    else:
        if name=='fc6':
            base_dict['0.weight']=checkpoint['backbone.fc6.weight']
    #         base_dict['0.bias']=checkpoint['backbone.fc6.bias']
    # import pdb
    # pdb.set_trace()
    state_dict.update(base_dict)
    net.load_state_dict(state_dict)
    print('model load pretrained weights')
    
class C3DBackbone(nn.Module):
    def __init__(self):
        super(C3DBackbone, self).__init__()
        # 112
        self.conv1a = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # 56
        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 28
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 14
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 7
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        # self.pool5=nn.AdaptiveAvgPool3d(1)
        # self.fc6 = nn.Linear(8192, 4096)

        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.relu(self.conv1a(x))
        x = self.pool1(x)
        x = self.relu(self.conv2a(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        out_4=x

        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        # x = self.pool5(x)

        # x = x.view(-1, 8192)
        # x = self.relu(self.fc6(x))
        return x,out_4
    
    
class C3D_Hard(nn.Module):
    def __init__(self, feature_dim, dropout_rate,freeze_backbone,freeze_blocks,pretrained_backbone=False,rgb_model_path=None, flow_model_path=None):
        super(C3D_Hard, self).__init__()

        self.rgb_backbone = C3DBackbone()
        self.flow_backbone = I3D(modality='flow')

        self.hard_sim_sampler = Hard_sim_sample_generator()
        self.hard_score_sampler = Hard_score_sample_generator()
        self.relation = VideoRelation(feat_dim=feature_dim)
        self.relation_two = VideoRelation(feat_dim=feature_dim)

        self.p_classifier_rgb = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(512, 2))
        self.p_classifier_flow = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(1024, 2))
        self.f_classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(512, 2))

        # self.Regressor=nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(512,2))

        self.GAP=nn.AdaptiveAvgPool3d(1)
        self.freeze_backbone = freeze_backbone

        if freeze_blocks==None:
            self.freeze_blocks=['conv1a','conv2a','conv3a','conv3b','conv4a','conv4b','conv5a','conv5b']
        else:
            self.freeze_blocks = freeze_blocks
        self.Softmax=nn.Softmax(dim=-1)

        if pretrained_backbone and rgb_model_path != None and flow_model_path != None:
            load_c3d_pretrained_model(self.rgb_backbone, rgb_model_path)

            flow_dict = torch.load(flow_model_path)
            flow_state_dict = self.flow_backbone.state_dict()
            flow_new_dict = {k: v for k, v in flow_dict.items() if k in flow_state_dict.keys()}
            flow_state_dict.update(flow_new_dict)
            self.flow_backbone.load_state_dict(flow_state_dict)

    def freeze_part_model(self):
        if self.freeze_backbone:
            for name,p in self.backbone.named_parameters():
                if name.split('.')[0] in self.freeze_blocks:
                    p.requires_grad=False

        else:
            for name,p in self.backbone.named_parameters():
                p.requires_grad=True


    # def train(self,mode=True):
    #     super(C3D_SGA_STD, self).train(mode)
    #     if self.freeze_backbone:
    #         self.freeze_part_model()
    #     return self

    def forward(self, ref_rgb, ref_flow, normal_rgb, normal_flow, abnormal_rgb, abnormal_flow, mode='normal',
                isTrain=True):
        # print(ref_rgb.shape)       # [B,N,C,T,H,W]  [4,3,(10crops),3,16,224,224]
        inputs_rgb = torch.cat((ref_rgb, normal_rgb, abnormal_rgb), 0)
        inputs_flow = torch.cat((ref_flow, normal_flow, abnormal_flow), 0)
        bs, N, C, T, H, W = inputs_rgb.size()
        bs_f, N_f, C_f, T_f, H_f, W_f = inputs_flow.size()
        inputs_rgb = inputs_rgb.view(bs*N, C, T, H, W)
        inputs_flow = inputs_flow.view(bs_f*N_f, C_f, T_f, H_f, W_f)

        inputs_rgb_feat, _ = self.rgb_backbone(inputs_rgb)
        inputs_rgb_feat = self.GAP(inputs_rgb_feat).squeeze(-1).squeeze(-1).squeeze(-1) # [B*N,F]
        with torch.no_grad():
            inputs_flow_feat, _ = self.flow_backbone(inputs_flow)
            inputs_flow_feat = self.GAP(inputs_flow_feat).squeeze(-1).squeeze(-1).squeeze(-1)

        F_len = inputs_rgb_feat.size()[-1]
        F_len_f = inputs_flow_feat.size()[-1]

        inputs_rgb_feat = inputs_rgb_feat.view(-1, N, F_len)
        inputs_flow_feat = inputs_flow_feat.view(-1, N, F_len_f)

        p_scores_rgb = self.Softmax(self.p_classifier_rgb(inputs_rgb_feat))[:, :, 1]  # (bs, N)
        p_scores_flow = self.Softmax(self.p_classifier_flow(inputs_flow_feat))[:, :, 1]

        bs = bs // 3
        ref_p_scores_rgb, abn_scores_rgb = p_scores_rgb[:bs], p_scores_rgb[2 * bs:3 * bs]
        ref_p_scores_flow, abn_scores_flow = p_scores_flow[:bs], p_scores_flow[2 * bs:3 * bs]

        ref_rgb, ref_flow, normal_rgb, normal_flow, abnormal_rgb, abnormal_flow = \
            inputs_rgb_feat[:bs], inputs_flow_feat[:bs], inputs_rgb_feat[bs:2 * bs], \
            inputs_flow_feat[bs:2 * bs], inputs_rgb_feat[2 * bs:3 * bs], inputs_flow_feat[2 * bs:3 * bs]

        if not isTrain:
            ref_aggr = self.relation(ref_rgb, ref_rgb, index=0)  # 测试时直接与自身进行融合
            ref_aggr = self.relation_two(ref_aggr, ref_aggr, index=0)
            ref_scores = self.Softmax(self.f_classifier(ref_aggr))[:,:,1]   #[bs, N]
            return ref_scores

        supn_hard_feat, supn_conf_feat = self.hard_sim_sampler(normal_rgb)
        supa_hard_nor_feat, supa_hard_abn_feat, supa_conf_nor_feat, supa_conf_abn_feat \
            = self.hard_score_sampler(abnormal_rgb, abn_scores_rgb, abn_scores_flow)

        # 第一步融合
        supn_aggr_feat = self.relation(supn_conf_feat, supn_hard_feat, index=0)  # 融合正常视频困难样本的特征，使得conf样本更具有鲁棒性
        supa_aggr_nor_feat = self.relation(supa_conf_nor_feat, supa_hard_nor_feat, index=0)  # 异常视频normal clips融合
        supa_aggr_abn_feat = self.relation(supa_conf_abn_feat, supa_hard_abn_feat, index=0)  # 异常视频abnormal clips融合

        if mode == 'normal':
            ref_hard_feat, ref_conf_feat = self.hard_sim_sampler(ref_rgb)
            ref_aggr_feat = self.relation(ref_conf_feat, ref_hard_feat, index=0)
            # 第二步融合
            # 融合不同视频正常clips
            ref_aggr2_feat = self.relation_two(ref_aggr_feat, torch.cat([supn_aggr_feat, supa_aggr_nor_feat], dim=1),
                                               index=0)
            # ref_aggr2_feat = torch.cat([ref_aggr_feat, torch.cat([supn_aggr_feat, supa_aggr_nor_feat], dim=1)], dim=1)

            # 对于异常clips第二步融合就自己与自己融，与正常clips的特征构建Rank loss
            supa_aggr2_abn_feat = self.relation_two(supa_aggr_abn_feat, supa_aggr_abn_feat, index=0)

            ref_scores = self.Softmax(self.f_classifier(ref_aggr2_feat))[:,:,1]

            return ref_p_scores_rgb, ref_p_scores_flow, ref_scores, ref_aggr2_feat, supa_aggr2_abn_feat

        else:
            ref_hard_nor_feat, ref_hard_abn_feat, ref_conf_nor_feat, ref_conf_abn_feat \
                = self.hard_score_sampler(ref_rgb, ref_p_scores_rgb, ref_p_scores_flow)
            ref_aggr_nor_feat = self.relation(ref_conf_nor_feat, ref_hard_nor_feat, index=0)
            ref_aggr_abn_feat = self.relation(ref_conf_abn_feat, ref_hard_abn_feat, index=0)

            # 第二步融合
            ref_aggr2_abn_feat = self.relation_two(ref_aggr_abn_feat, supa_aggr_abn_feat, index=0)
            # ref_aggr2_abn_feat = torch.cat([ref_aggr_abn_feat, supa_aggr_abn_feat], dim=1)

            ref_aggr2_nor_feat = self.relation_two(ref_aggr_nor_feat,
                                                   torch.cat([supn_aggr_feat, supa_aggr_nor_feat], dim=1), index=0)

            ref_scores = self.Softmax(self.f_classifier(ref_aggr2_abn_feat))[:,:,1]

            return ref_p_scores_rgb, ref_p_scores_flow, ref_scores, ref_aggr2_nor_feat, ref_aggr2_abn_feat

    # def forward(self,x):
    #     feat_map,feat_map_4=self.backbone(x)
    #
    #     feat=self.GAP(feat_map).squeeze(-1).squeeze(-1).squeeze(-1)
    #     # feat=self.pool5(feat_map).view(-1,8192)
    #     # feat=self.fc6(feat)
    #
    #     logits=self.Regressor(feat)
    #
    #     return logits
