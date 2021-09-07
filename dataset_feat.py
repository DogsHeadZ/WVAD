import torch.utils.data as data
import numpy as np
import torch
import numpy as np
import h5py
import cv2
import os

from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def process_feat(feat, length):  # 在训练时用到这个函数，因为预训练的特征是等间隔16划分的即T是不固定的，这里是为了将T固定为32（通过合并特征）
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


class Dataset_f(data.Dataset):
    def __init__(self, rgb_list_file, rgb_file, flow_file, train_txt, test_txt, test_mask_dir, is_normal=True, segment_len=16, transform=None, test_mode=False):
        self.rgb_list_file = rgb_list_file
        self.rgb_file = rgb_file
        self.flow_file = flow_file
        self.train_txt = train_txt
        self.test_txt = test_txt
        self.test_mask_dir = test_mask_dir
        self.is_normal = is_normal
        self.segment_len = segment_len
        self.tranform = transform
        self.test_mode = test_mode
        if not test_mode:
            self.get_vid_names_dict()

            if is_normal:
                self.selected_keys = list(self.norm_vid_names_dict.keys())
                self.selected_dict = self.norm_vid_names_dict
            else:
                self.selected_keys = list(self.abnorm_vid_names_dict.keys())
                self.selected_dict = self.abnorm_vid_names_dict

        else:
            self.test_dict_annotation()
            self.selected_keys = list(self.annotation_dict.keys())
            self.selected_dict = self.annotation_dict

        self._parse_list()

    def get_vid_names_dict(self):
        keys = sorted(list(h5py.File(self.rgb_file, 'r').keys()))
        self.norm_vid_names_dict = {}
        self.abnorm_vid_names_dict = {}

        for line in open(self.train_txt,'r').readlines():
            key,label=line.strip().split(',')
            if label=='1':
                for k in keys:
                    if key == k.split('-')[0]:
                        if key in self.abnorm_vid_names_dict.keys():
                            self.abnorm_vid_names_dict[key]+=1
                        else:
                            self.abnorm_vid_names_dict[key]=1
            else:
                for k in keys:
                    if key == k.split('-')[0]:
                        if key in self.norm_vid_names_dict.keys():
                            self.norm_vid_names_dict[key]+=1
                        else:
                            self.norm_vid_names_dict[key]=1

    def test_dict_annotation(self):
        self.annotation_dict = {}
        keys=sorted(list(h5py.File(self.rgb_file, 'r').keys()))
        for line in open(self.test_txt,'r').readlines():
            key,anno_type,frames_num = line.strip().split(',') # 这里的framenum是错的
            frames_seg_num = 0
            for k in keys:
                if k.split('-')[0] == key:
                    frames_seg_num += 1
            if anno_type=='1':
                label='Abnormal'
                anno = np.load(os.path.join(self.test_mask_dir, key + '.npy'))[:frames_seg_num * self.segment_len]
            else:
                label='Normal'
                anno=np.zeros(frames_seg_num * self.segment_len,dtype=np.uint8)
            self.annotation_dict[key]=[anno,frames_seg_num]

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.is_normal:
                self.list = self.list[63:]
                print('normal list')
                # print(self.list)
            else:
                self.list = self.list[:63]
                print('abnormal list')
                # print(self.list)

    def __getitem__(self, index):
        # 这里是加载RTFM给的特征
        label = self.get_label(index)  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            frames_seg_num = features.shape[0]
            key = self.list[index].strip('\n').split('/')[-1].split('.')[0][:-4]
            if index<44:
                anno = np.load(os.path.join(self.test_mask_dir, key + '.npy'))[:frames_seg_num * self.segment_len]
                if anno.shape[0] < frames_seg_num * self.segment_len:
                    anno = anno[:(frames_seg_num-1) * self.segment_len]
                    features = features[:-1]
            else:
                anno=np.zeros(frames_seg_num * self.segment_len,dtype=np.uint8)
            return features, features, anno
        else:
            features = features.transpose(1, 0, 2)  # [10, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, divided_features, label

        ##########################   #这里是加载我们提取的i3d特征，效果很差，就没用下面这些代码，但暂且保留
        key = self.selected_keys[index]
        if not self.test_mode:
            video_len = self.selected_dict[key]
        else:
            video_len = self.selected_dict[key][1]
        frame_feats = []
        flow_feats = []
        with h5py.File(self.rgb_file, 'r') as rgb_h5, h5py.File(self.flow_file, 'r') as flow_h5:
            for i in range(video_len):
                frame_feats.extend(rgb_h5[key + '-{0:06d}'.format(i)][:])  # [1,1024]
                flow_feats.extend(flow_h5[key + '-{0:06d}'.format(i)][:])
        frame_feats = np.stack(frame_feats)
        flow_feats = np.stack(flow_feats)

        label = self.get_label(index) # get video level label 0/1

        if self.tranform is not None:
            frame_feats = self.tranform(frame_feats)
            flow_feats = self.tranform(flow_feats)

        if self.test_mode:

            anno = self.annotation_dict[key][0]
            return torch.from_numpy(frame_feats).unsqueeze(0), torch.from_numpy(flow_feats).unsqueeze(0), torch.from_numpy(anno)
        else:
            frame_feats = process_feat(frame_feats, 32)  # [32, F]
            frame_feats = np.array(frame_feats, dtype=np.float32)

            flow_feats = process_feat(flow_feats, 32)  # [32, F]
            flow_feats = np.array(flow_feats, dtype=np.float32)

            return torch.from_numpy(frame_feats).unsqueeze(0), torch.from_numpy(flow_feats).unsqueeze(0), torch.from_numpy(label)

    def get_label(self, index):
        if self.is_normal:
            # label[0] = 1
            label = np.zeros(1, dtype=np.float32)
        else:
            label = np.ones(1, dtype=np.float32)
            # label[1] = 1
        return label

    def __len__(self):
        return len(self.selected_keys)
