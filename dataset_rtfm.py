import torch.utils.data as data
import numpy as np
import torch
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


class Dataset(data.Dataset):
    def __init__(self, rgb_list_file, is_normal=True, transform=None, test_mode=False):
        self.is_normal = is_normal
        self.rgb_list_file = rgb_list_file

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.is_normal:
                self.list = self.list[63:]
                print('normal list')
                print(self.list)
            else:
                self.list = self.list[:63]

                print('abnormal list')
                print(self.list)

    def __getitem__(self, index):

        label = self.get_label(index) # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            features = features.transpose(1, 0, 2)  # [10, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self, index):
        if self.is_normal:
            # label[0] = 1
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
            # label[1] = 1
        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
