import math
import os
import torch
import torch.nn as nn
import numpy as np
from model.modules import Fusion, Memory, SA

def get_padding_shape(filter_shape, stride, mod=0):
    """Fetch a tuple describing the input padding shape.
    NOTES: To replicate "TF SAME" style padding, the padding shape needs to be
    determined at runtime to handle cases when the input dimension is not divisible
    by the stride.
    See https://stackoverflow.com/a/49842071 for explanation of TF SAME padding logic
    """
    def _pad_top_bottom(filter_dim, stride_val, mod):
        if mod:
            pad_along = max(filter_dim - mod, 0)
        else:
            pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for idx, (filter_dim, stride_val) in enumerate(zip(filter_shape, stride)):
        depth_mod = (idx == 0) and mod
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val, depth_mod)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)

    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)
    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        self.stride = stride
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
            if stride[0] > 1:
                padding_shapes = [get_padding_shape(kernel_size, stride, mod) for
                                  mod in range(stride[0])]
            else:
                padding_shapes = [padding_shape]
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pads = [torch.nn.ConstantPad3d(x, 0) for x in padding_shapes]
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            # This is not strictly the correct map between epsilons in keras and
            # pytorch (which have slightly different definitions of the batch norm
            # forward pass), but it seems to be good enough. The PyTorch formula
            # is described here:
            # https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html
            tf_style_eps = 1E-3
            self.batch3d = torch.nn.BatchNorm3d(out_channels, eps=tf_style_eps)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            # Determine the padding to be applied by examining the input shape
            pad_idx = inp.shape[2] % self.stride[0]
            pad_op = self.pads[pad_idx]
            inp = pad_op(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.stride = stride
            if stride[0] > 1:
                padding_shapes = [get_padding_shape(kernel_size, stride, mod) for
                                  mod in range(stride[0])]
            else:
                padding_shapes = [padding_shape]
            self.pads = [torch.nn.ConstantPad3d(x, 0) for x in padding_shapes]
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        pad_idx = inp.shape[2] % self.stride[0]
        pad_op = self.pads[pad_idx]
        inp = pad_op(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3D(torch.nn.Module):
    def __init__(self,
                 modality='rgb'):
        super(I3D, self).__init__()

        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        self.conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        # self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        # self.dropout = torch.nn.Dropout(dropout_prob)
        # self.conv3d_0c_1x1 = Unit3Dpy(
        #     in_channels=1024,
        #     out_channels=self.num_classes,
        #     kernel_size=(1, 1, 1),
        #     activation=None,
        #     use_bias=True,
        #     use_bn=False)
        # self.softmax = torch.nn.Softmax(1)

    def forward(self, inp):
        # Preprocessing
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out_3f=out
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out_4f=out
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)

        # out = self.avg_pool(out)
        # out = self.dropout(out)
        # out = self.conv3d_0c_1x1(out)
        # out = out.squeeze(3)
        # out = out.squeeze(3)
        # out = out.mean(2)     # ?????????????????????????????????16???????????????
        # out_logits = out
        # out = self.softmax(out_logits)
        return out,out_4f#, out_logits


class I3D_Base(nn.Module):
    def __init__(self,dropout_rate,expand_k,freeze_bn=False,freeze_backbone=False,freeze_blocks=None,freeze_bn_statics=False,
                 pretrained_backbone=False,rgb_model_path=None,flow_model_path=None):
        super(I3D_Base, self).__init__()
        self.rgb_backbone=I3D(modality='rgb')
        self.flow_backbone=I3D(modality='flow')

        self.Regressor=nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(1024,2))
        self.Softmax = nn.Softmax(dim=-1)

        self.freeze_bn=freeze_bn
        self.freeze_backbone=freeze_backbone
        self.rgb_GAP=nn.AdaptiveAvgPool3d(1)
        self.flow_GAP=nn.AdaptiveAvgPool3d(1)

        self.freeze_bn_statics = freeze_bn_statics
        if freeze_blocks==None:
            self.freeze_blocks=['conv3d_1a_7x7','conv3d_2b_1x1','conv3d_2c_3x3','mixed_3b','mixed_3c','mixed_4b',
                                'mixed_4c','mixed_4d','mixed_4e','mixed_4f','mixed_5b','mixed_5c']
        else:
            self.freeze_blocks=freeze_blocks
        if pretrained_backbone and rgb_model_path!=None and flow_model_path!=None:
            self.load_part_model(rgb_model_path, flow_model_path)


    def load_part_model(self, rgb_model_path, flow_model_path):
        rgb_dict=torch.load(rgb_model_path)
        rgb_state_dict=self.rgb_backbone.state_dict()
        rgb_new_dict={k:v for k,v in rgb_dict.items() if k in rgb_state_dict.keys()}
        rgb_state_dict.update(rgb_new_dict)
        self.rgb_backbone.load_state_dict(rgb_state_dict)

        flow_dict = torch.load(flow_model_path)
        flow_state_dict = self.flow_backbone.state_dict()
        flow_new_dict = {k: v for k, v in flow_dict.items() if k in flow_state_dict.keys()}
        flow_state_dict.update(flow_new_dict)
        self.flow_backbone.load_state_dict(flow_state_dict)

    def freeze_part_model(self):
        if self.freeze_backbone:
            for name,p in self.rgb_backbone.named_parameters():
                if name.split('.')[0] in self.freeze_blocks:
                    p.requires_grad=False
            for name,p in self.flow_backbone.named_parameters():
                if name.split('.')[0] in self.freeze_blocks:
                    p.requires_grad=False

        else:
            for name,p in self.rgb_backbone.named_parameters():
                if name.split('.')[0] in self.freeze_blocks:
                    p.requires_grad=True
            for name,p in self.flow_backbone.named_parameters():
                if name.split('.')[0] in self.freeze_blocks:
                    p.requires_grad=True

    def freeze_batch_norm(self):
        if self.freeze_bn:
            for name, module in self.rgb_backbone.named_modules():
                if isinstance(module,nn.BatchNorm3d) or isinstance(module,nn.BatchNorm2d):
                    if name.split('.')[0] in self.freeze_blocks:
                        if self.freeze_bn_statics:    # ???????????????????????????????????????????????????????????????False?????????
                            module.eval()
                        else:
                            module.train()
                        module.weight.requires_grad=False
                        module.bias.requires_grad=False

            for name, module in self.flow_backbone.named_modules():
                if isinstance(module,nn.BatchNorm3d) or isinstance(module,nn.BatchNorm2d):
                    if name.split('.')[0] in self.freeze_blocks:
                        if self.freeze_bn_statics:    # ???????????????????????????????????????????????????????????????False?????????
                            module.eval()
                        else:
                            module.train()
                        module.weight.requires_grad=False
                        module.bias.requires_grad=False

        else:
            for name, module in self.rgb_backbone.named_modules():
                if isinstance(module,nn.BatchNorm3d) or isinstance(module,nn.BatchNorm2d):
                    if name.split('.')[0] in self.freeze_blocks:
                        if self.freeze_bn_statics:
                            module.eval()
                        else:
                            module.train()
                        module.weight.requires_grad=True
                        module.bias.requires_grad=True

            for name, module in self.flow_backbone.named_modules():
                if isinstance(module,nn.BatchNorm3d) or isinstance(module,nn.BatchNorm2d):
                    if name.split('.')[0] in self.freeze_blocks:
                        if self.freeze_bn_statics:
                            module.eval()
                        else:
                            module.train()
                        module.weight.requires_grad=True
                        module.bias.requires_grad=True

    def train(self,mode=True):
        super(I3D_Base, self).train(mode)
        if self.freeze_backbone:
            self.freeze_part_model()
        if self.freeze_bn:
            self.freeze_batch_norm()
        return self

    def forward(self,x_rgb, x_flow):
        # print(x_rgb.shape)

        rgb_feat_map, rgb_feat_map_4f = self.rgb_backbone(x_rgb)
        # flow_feat_map, flow_feat_map_4f = self.flow_backbone(x_flow)

        rgb_feat=self.rgb_GAP(rgb_feat_map).squeeze(-1).squeeze(-1).squeeze(-1)
        rgb_logits=self.Softmax(self.Regressor(rgb_feat))

        # flow_feat = self.flow_GAP(flow_feat_map).squeeze(-1).squeeze(-1).squeeze(-1)
        # flow_logits = self.Softmax(self.Regressor(flow_feat))
        #
        # out_logits = rgb_logits + flow_logits
        # out_logits = self.Softmax(out_logits)

        return rgb_logits


class I3D_Co(nn.Module):

    def __init__(self,dropout_rate,expand_k,freeze_bn=False,freeze_backbone=False,freeze_blocks=None,freeze_bn_statics=False,
                 pretrained_backbone=False,rgb_model_path=None,flow_model_path=None):
        super(I3D_Co, self).__init__()
        self.rgb_backbone=I3D(modality='rgb')
        self.flow_backbone=I3D(modality='flow')
        self.pl_generator = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(1024,2))
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(1024,2))
        self.Softmax = nn.Softmax(dim=-1)

        self.freeze_bn=freeze_bn
        self.freeze_backbone=freeze_backbone
        self.freeze_bn_statics = freeze_bn_statics

        self.rgb_GAP=nn.AdaptiveAvgPool3d(1)
        self.flow_GAP=nn.AdaptiveAvgPool3d(1)

        self.fusion = Fusion()
        self.memory = Memory()
        self.SA = SA()

        if freeze_blocks==None:
            self.freeze_blocks=['conv3d_1a_7x7','conv3d_2b_1x1','conv3d_2c_3x3','mixed_3b','mixed_3c','mixed_4b',
                                'mixed_4c','mixed_4d','mixed_4e','mixed_4f','mixed_5b','mixed_5c']
        else:
            self.freeze_blocks=freeze_blocks

        if pretrained_backbone and rgb_model_path!=None and flow_model_path!=None:
            self.load_part_model(rgb_model_path, flow_model_path)

    def load_part_model(self, rgb_model_path, flow_model_path):
        rgb_dict=torch.load(rgb_model_path)
        rgb_state_dict=self.rgb_backbone.state_dict()
        rgb_new_dict={k:v for k,v in rgb_dict.items() if k in rgb_state_dict.keys()}
        rgb_state_dict.update(rgb_new_dict)
        self.rgb_backbone.load_state_dict(rgb_state_dict)

        flow_dict = torch.load(flow_model_path)
        flow_state_dict = self.flow_backbone.state_dict()
        flow_new_dict = {k: v for k, v in flow_dict.items() if k in flow_state_dict.keys()}
        flow_state_dict.update(flow_new_dict)
        self.flow_backbone.load_state_dict(flow_state_dict)

        # moodel_dict = torch.load(rgb_model_path)['model']
        # rgb_state_dict = self.rgb_backbone.state_dict()
        # rgb_new_dict = {k[20:]: v for k, v in moodel_dict.items() if k[20:] in rgb_state_dict.keys()}  #????????????'module.rgb_backbone.'
        # rgb_state_dict.update(rgb_new_dict)
        # self.rgb_backbone.load_state_dict(rgb_state_dict)
        #
        # flow_state_dict = self.flow_backbone.state_dict()
        # flow_new_dict = {k[21:]: v for k, v in moodel_dict.items() if k[21:] in flow_state_dict.keys()} #????????????'module.flow_backbone.'
        # flow_state_dict.update(flow_new_dict)
        # self.flow_backbone.load_state_dict(flow_state_dict)
        #
        # generator_state_dict = self.pl_generator.state_dict()
        # generator_new_dict = {k[17:]: v for k, v in moodel_dict.items() if k[17:] in flow_state_dict.keys()} #????????????'module.Regressor.'
        # generator_state_dict.update(generator_new_dict)
        # self.pl_generator.load_state_dict(generator_state_dict)
        # self.classifier.load_state_dict(generator_state_dict)


    def freeze_part_model(self):
        if self.freeze_backbone:
            for name,p in self.rgb_backbone.named_parameters():
                if name.split('.')[0] in self.freeze_blocks:
                    p.requires_grad=False
            for name,p in self.flow_backbone.named_parameters():
                if name.split('.')[0] in self.freeze_blocks:
                    p.requires_grad=False

        else:
            for name,p in self.rgb_backbone.named_parameters():
                if name.split('.')[0] in self.freeze_blocks:
                    p.requires_grad=True
            for name,p in self.flow_backbone.named_parameters():
                if name.split('.')[0] in self.freeze_blocks:
                    p.requires_grad=True

    def freeze_batch_norm(self):
        if self.freeze_bn:
            for name, module in self.rgb_backbone.named_modules():
                if isinstance(module,nn.BatchNorm3d) or isinstance(module,nn.BatchNorm2d):
                    if name.split('.')[0] in self.freeze_blocks:
                        if self.freeze_bn_statics:    # ???????????????????????????????????????????????????????????????False?????????
                            module.eval()
                        else:
                            module.train()
                        module.weight.requires_grad=False
                        module.bias.requires_grad=False

            for name, module in self.flow_backbone.named_modules():
                if isinstance(module,nn.BatchNorm3d) or isinstance(module,nn.BatchNorm2d):
                    if name.split('.')[0] in self.freeze_blocks:
                        if self.freeze_bn_statics:    # ???????????????????????????????????????????????????????????????False?????????
                            module.eval()
                        else:
                            module.train()
                        module.weight.requires_grad=False
                        module.bias.requires_grad=False

        else:
            for name, module in self.rgb_backbone.named_modules():
                if isinstance(module,nn.BatchNorm3d) or isinstance(module,nn.BatchNorm2d):
                    if name.split('.')[0] in self.freeze_blocks:
                        if self.freeze_bn_statics:
                            module.eval()
                        else:
                            module.train()
                        module.weight.requires_grad=True
                        module.bias.requires_grad=True

            for name, module in self.flow_backbone.named_modules():
                if isinstance(module,nn.BatchNorm3d) or isinstance(module,nn.BatchNorm2d):
                    if name.split('.')[0] in self.freeze_blocks:
                        if self.freeze_bn_statics:
                            module.eval()
                        else:
                            module.train()
                        module.weight.requires_grad=True
                        module.bias.requires_grad=True

    def train(self,mode=True):
        super(I3D_Co, self).train(mode)
        if self.freeze_backbone:
            self.freeze_part_model()
        if self.freeze_bn:
            self.freeze_batch_norm()
        return self

    def forward_test(self, ref_rgb):
        ref_rgb_feat_map, ref_rgb_feat_map_4f = self.rgb_backbone(ref_rgb)
        ref_rgb_feat = self.rgb_GAP(ref_rgb_feat_map).squeeze(-1).squeeze(-1).squeeze(-1)  # [B,F]
        ref_attn_feat = self.SA(ref_rgb_feat)

        xxx = self.memory(ref_attn_feat)  # todo

        scores = self.Softmax(self.classifier(xxx))
        return scores


    def forward(self,ref_rgb, ref_flow, normal_rgb, normal_flow, abnormal_rgb, abnormal_flow, mode='normal'):
        # ref_rgb: [B,N,C,T,H,W] N??????????????????clip????????????B???batchsize????????????????????????

        bs, N, C, T, H, W = ref_rgb.shape
        ref_rgb = ref_rgb.view([-1, C, T, H, W])
        ref_flow = ref_flow.view([-1, 2, T, H, W])
        normal_rgb = normal_rgb.view([-1, C, T, H, W])
        normal_flow = normal_flow.view([-1, 2, T, H, W])
        abnormal_rgb = abnormal_rgb.view([-1, C, T, H, W])
        abnormal_flow = abnormal_flow.view([-1, 2, T, H, W])

        ref_rgb_feat_map, ref_rgb_feat_map_4f = self.rgb_backbone(ref_rgb)
        ref_rgb_feat = self.rgb_GAP(ref_rgb_feat_map).squeeze(-1).squeeze(-1).squeeze(-1)   #[B*N,F]
        with torch.no_grad():
            ref_flow_feat_map, ref_flow_feat_map_4f = self.flow_backbone(ref_flow)
            ref_flow_feat = self.flow_GAP(ref_flow_feat_map).squeeze(-1).squeeze(-1).squeeze(-1)
        ref_p_score=self.Softmax(self.pl_generator(ref_rgb_feat))

        nor_rgb_feat_map, nor_rgb_feat_map_4f = self.rgb_backbone(normal_rgb)
        nor_rgb_feat = self.rgb_GAP(nor_rgb_feat_map).squeeze(-1).squeeze(-1).squeeze(-1)
        with torch.no_grad():
            nor_flow_feat_map, nor_flow_feat_map_4f = self.flow_backbone(normal_flow)
            nor_flow_feat = self.flow_GAP(nor_flow_feat_map).squeeze(-1).squeeze(-1).squeeze(-1)

        abn_rgb_feat_map, abn_rgb_feat_map_4f = self.rgb_backbone(abnormal_rgb)
        abn_rgb_feat = self.rgb_GAP(abn_rgb_feat_map).squeeze(-1).squeeze(-1).squeeze(-1)
        with torch.no_grad():
            abn_flow_feat_map, abn_flow_feat_map_4f = self.flow_backbone(abnormal_flow)
            abn_flow_feat = self.flow_GAP(abn_flow_feat_map).squeeze(-1).squeeze(-1).squeeze(-1)
        abn_score=self.Softmax(self.pl_generator(ref_rgb_feat))

        F = ref_rgb_feat.shape[-1]
        ref_rgb_feat = ref_rgb_feat.view([bs, N, F])
        ref_flow_feat = ref_flow_feat.view([bs, N, F])
        nor_rgb_feat = nor_rgb_feat.view([bs, N, F])
        nor_flow_feat = nor_flow_feat.view([bs, N, F])
        abn_rgb_feat = abn_rgb_feat.view([bs, N, F])
        abn_flow_feat = abn_flow_feat.view([bs, N, F])
        abn_score = abn_score.view([bs, N, 2])[:, :, -1]
        ref_p_score = ref_p_score.view([bs, N, 2])[:, :, -1]

        if mode == 'normal':
            # ???ref???normal???
            nor_topK = 1   #???abnormal video??????1???clip??????normal
            abn_k_idx_nor = torch.topk(abn_score, nor_topK, dim=1, largest=False)[1]
            abn_k_idx_nor = abn_k_idx_nor.unsqueeze(2).expand([-1, -1, F])
            abn_rgb_feat_nor = torch.gather(abn_rgb_feat, 1, abn_k_idx_nor)
            abn_flow_feat_nor = torch.gather(abn_flow_feat, 1, abn_k_idx_nor)

            abn_topK = 3   #?????????abnormal video???3???clip??????abnormal
            abn_k_idx_abn = torch.topk(abn_score, abn_topK, dim=1, largest=True)[1]
            abn_k_idx_abn = abn_k_idx_abn.unsqueeze(2).expand([-1, -1, F])
            abn_rgb_feat_abn = torch.gather(abn_rgb_feat, 1, abn_k_idx_abn)
            abn_flow_feat_abn = torch.gather(abn_flow_feat, 1, abn_k_idx_abn)

            nor_rgb_feat = torch.cat([nor_rgb_feat, abn_rgb_feat_nor], dim=1)  # ????????????????????????????????????
            nor_flow_feat = torch.cat([nor_flow_feat, abn_flow_feat_nor], dim=1)

            fusion_feat = self.fusion(ref_rgb_feat, ref_flow_feat, nor_rgb_feat, nor_flow_feat)
            ref_attn_feat = self.SA(fusion_feat)
            sup_attn_feat = self.SA(abn_rgb_feat_abn) # for rank loss
        else:
            # ???ref???abnormal???
            ref_topK = 4
            ref_k_idx_abn = torch.topk(ref_p_score, ref_topK, dim=1, largest=True)[1]
            ref_k_idx_abn = ref_k_idx_abn.unsqueeze(2).expand([-1, -1, F])
            ref_rgb_feat_abn = torch.gather(ref_rgb_feat, 1, ref_k_idx_abn)
            ref_flow_feat_abn = torch.gather(ref_flow_feat, 1, ref_k_idx_abn)

            abn_topK = 4
            abn_k_idx_abn = torch.topk(abn_score, abn_topK, dim=1, largest=True)[1]
            abn_k_idx_abn = abn_k_idx_abn.unsqueeze(2).expand([-1, -1, F])
            abn_rgb_feat_abn = torch.gather(abn_rgb_feat, 1, abn_k_idx_abn)
            abn_flow_feat_abn = torch.gather(abn_flow_feat, 1, abn_k_idx_abn)

            fusion_feat_nor = self.fusion(ref_rgb_feat_abn, ref_flow_feat_abn, abn_rgb_feat_abn, abn_flow_feat_abn)
            ref_attn_feat = self.SA(fusion_feat_nor)
            sup_attn_feat = self.SA(nor_rgb_feat)  # for rank loss

        xxx = self.memory(ref_attn_feat)  #todo

        score = self.Softmax(self.classifier(xxx))

        return score, ref_p_score, ref_attn_feat, sup_attn_feat
