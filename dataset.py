import torch
from torch.utils.data import Dataset
import torchvision
from opencv_videovision import transforms
import numpy as np
import h5py
import cv2
import os
from train_utils import random_perturb
import time

Abnormal_type=['Abuse','Arrest','Arson','Assault','Burglary',
               'Explosion','Fighting','RoadAccidents','Robbery',
               'Shooting','Shoplifting','Stealing','Vandalism','Normal']

class Test_Dataset_SHT_I3D(Dataset):
    def __init__(self,rgb_h5_file, flow_h5_file, test_txt, test_mask_dir,
                 segment_len=16, ten_crop=False, height=256, width=340, crop_size=224):
        self.rgb_h5_file = rgb_h5_file
        self.flow_h5_file = flow_h5_file

        # self.keys = sorted(list(h5py.File(self.h5_path, 'r').keys()))
        self.test_txt=test_txt
        self.segment_len = segment_len
        self.ten_crop = ten_crop
        self.crop_size = crop_size
        self.test_mask_dir=test_mask_dir

        self.mean=[128,128,128]
        self.std=[128,128,128]
        self.height=height
        self.width=width

        self.test_dict_annotation()

        if ten_crop:
            self.transforms = transforms.Compose([transforms.Resize([240, 320]),
                                                    transforms.ClipToTensor(div_255=False),
                                                    transforms.Normalize(mean=self.mean,std=self.std),
                                                    transforms.TenCropTensor(224)])

        else:
            self.rgb_transforms = transforms.Compose([ transforms.Resize([240, 320]),
                                                   # transforms.CenterCrop(self.crop_size),
                                                    transforms.ClipToTensor(div_255=False),
                                                    transforms.Normalize(mean=self.mean, std=self.std)])

            self.flow_transforms = transforms.Compose([transforms.Resize((256, 256)),  # TODO：这里也要改成原图大小
                                                       # transforms.MultiScaleCrop(224, [1.0, 0.8], max_distort=1,
                                                       #                           fix_crop=True),
                                                       transforms.ClipToTensor(channel_nb=2, div_255=False),
                                                       ])


        self.dataset_len = len(h5py.File(self.rgb_h5_file,'r')[self.keys[0]][:])

    def __len__(self):
        return len(self.keys)

    def test_dict_annotation(self):
        self.annotation_dict = {}
        self.keys=[]
        keys=sorted(list(h5py.File(self.rgb_h5_file, 'r').keys()))
        for line in open(self.test_txt,'r').readlines():
            key,anno_type,frames_num = line.strip().split(',')
            frames_num=int(frames_num)
            if anno_type=='1':
                label='Abnormal'
                anno = np.load(os.path.join(self.test_mask_dir, key + '.npy'))#[
                       #:frames_num - frames_num % self.segment_len]
            else:
                label='Normal'
                anno=np.zeros(frames_num-frames_num % self.segment_len,dtype=np.uint8)
            self.annotation_dict[key]=[anno,label]
        # key_dict={}
        for key in keys:
            if key.split('-')[0] in self.annotation_dict.keys():
                self.keys.append(key)

    def frame_processing(self, frames):
        new_frames = []
        for frame in frames:
            img_decode = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            new_frames.append(img_decode)
        new_frames = self.rgb_transforms(new_frames)
        return new_frames

    def flow_processing(self, flows):
        # flows = np.reshape(np.asarray(flows), (len(flows), 256, 256, 2)).astype(float)
        flows = np.asarray(flows).astype(float)
        flows = ((2 * (flows - flows.min()) / (flows.max() - flows.min())) - 1)  # Todo:源代码应该是整个视频而不是一段视频进行norm
        flows = self.flow_transforms(flows)
        return flows

    def __getitem__(self, i):
        key = self.keys[i]
        # frames=h5py.File(self.h5_path,'r')[key][:]
        with h5py.File(self.rgb_h5_file, 'r') as rgb_h5, h5py.File(self.flow_h5_file, 'r') as flow_h5:
            frames = rgb_h5[key][:]
            flows = flow_h5[key][:]

        key_tmp, idx = key.split('-')
        idx = int(idx)
        ano_type=self.annotation_dict[key_tmp][1]
        if ano_type=='Normal':
            anno=np.zeros([self.segment_len],dtype=np.uint8)
        else:
            anno = self.annotation_dict[key_tmp][0][
                   idx * self.segment_len :(idx + 1) * self.segment_len ].astype(np.uint8)

        # begin=time.time()
        frames = self.frame_processing(frames)
        flows = self.flow_processing(flows)

        if self.ten_crop:
            frames = torch.stack(frames)
        # end=time.time()
        # print('take time {}'.format(end-begin))
        return frames, flows, ano_type, idx, anno


class Train_TemAug_Dataset_SHT_I3D(Dataset):
    def __init__(self, rgb_h5_file, flow_h5_file ,train_txt, pseudo_labels,clip_num=8, segment_len=16,
                 type='Normal', ten_crop=False, rgb_diff=False,hard_label=False,score_segment_len=16,continuous_sampling=False):
        self.rgb_h5_file = rgb_h5_file
        self.flow_h5_file = flow_h5_file
        self.pseudo_labels = np.load(pseudo_labels, allow_pickle=True).tolist()
        self.keys = sorted(list(h5py.File(self.rgb_h5_file, 'r').keys()))
        self.clip_num=clip_num
        self.dataset_len = len(h5py.File(self.rgb_h5_file,'r')[self.keys[0]][:])
        self.segment_len = segment_len
        self.ten_crop = ten_crop
        self.rgb_diff=rgb_diff
        self.hard_label=hard_label
        self.score_segment_len=score_segment_len
        self.continuous_sampling=continuous_sampling

        self.train_txt=train_txt
        self.test_mask_dir = 'data/test_frame_mask/'

        # self.mean =torch.from_numpy(np.load('/mnt/sdd/jiachang/c3d_train01_16_128_171_mean.npy'))
        self.mean=[128,128,128]
        self.std=[128,128,128]
        self.get_vid_names_dict()
        self.type = type

        self.test_dict_annotation()

        if self.type == 'Normal':
            self.selected_keys = list(self.norm_vid_names_dict.keys())
            self.selected_dict=self.norm_vid_names_dict
        else:
            self.selected_keys = list(self.abnorm_vid_names_dict.keys())
            self.selected_dict=self.abnorm_vid_names_dict

        if self.ten_crop:
            # torchvision.transforms.TenCrop
            self.transforms = transforms.Compose([transforms.Resize([256,340]),
                                                    transforms.ClipToTensor(div_255=False),
                                                    transforms.Normalize(mean=self.mean, std=self.std),
                                                    transforms.TenCropTensor(224)])
        # torchvision.transforms.TenCrop(size)
        else:
            self.rgb_transforms=transforms.Compose([transforms.Resize((256,340)),
                                                # transforms.RandomCrop((112,112)),
                                                transforms.MultiScaleCrop(224, [1.0, 0.8], max_distort=1, fix_crop=True),
                                                transforms.RandomHorizontalFlip(),
                                                # transforms.RandomGrayScale(),
                                                transforms.ClipToTensor(div_255=False),
                                                transforms.Normalize(self.mean,self.std)
                                                ])

            self.flow_transforms = transforms.Compose([transforms.Resize((256, 256)),   # TODO：这里也要改成原图大小
                                                      # transforms.RandomCrop((112,112)),
                                                      transforms.MultiScaleCrop(224, [1.0, 0.8], max_distort=1,
                                                                                fix_crop=True),
                                                      transforms.ClipToTensor(channel_nb=2, div_255=False),
                                                      ])

    def __len__(self):
        return len(self.selected_keys)

    def test_dict_annotation(self):
        self.annotation_dict = {}
        self.keys=[]
        keys=sorted(list(h5py.File(self.rgb_h5_file, 'r').keys()))
        for line in open(self.train_txt,'r').readlines():
            key,anno_type = line.strip().split(',')

            if anno_type=='1':
                label='Abnormal'
                anno = np.load(os.path.join(self.test_mask_dir, key + '.npy'))#[
                       #:frames_num - frames_num % self.segment_len]
            # else:
            #     label='Normal'
            #     anno=np.zeros(len(anno)-len(anno) % self.segment_len,dtype=np.uint8)

                self.annotation_dict[key]=[anno,label]
        # key_dict={}
        for key in keys:
            if key.split('-')[0] in self.annotation_dict.keys():
                self.keys.append(key)

    def get_abnorm_mean(self):
        scores=0
        nums=0
        for key in self.abnorm_vid_names_dict:
            scores+=np.sum(self.pseudo_labels[key+'.npy'])
            nums+=self.pseudo_labels[key + '.npy'].shape[0]
        print(scores/nums)

    def get_vid_names_dict(self):
        self.norm_vid_names_dict = {}
        self.abnorm_vid_names_dict = {}

        for line in open(self.train_txt,'r').readlines():
            key,label=line.strip().split(',')
            if label=='1':
                for k in self.keys:
                    if key == k.split('-')[0]:
                        if key in self.abnorm_vid_names_dict.keys():
                            self.abnorm_vid_names_dict[key]+=1
                        else:
                            self.abnorm_vid_names_dict[key]=1
            else:
                for k in self.keys:
                    if key == k.split('-')[0]:
                        if key in self.norm_vid_names_dict.keys():
                            self.norm_vid_names_dict[key]+=1
                        else:
                            self.norm_vid_names_dict[key]=1

    def frame_processing(self, frames):
        new_frames = []
        for frame in frames:
            img_decode=cv2.cvtColor(cv2.imdecode(np.frombuffer(frame,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            new_frames.append(img_decode)
        del frames
        new_frames=self.rgb_transforms(new_frames)
        # new_frames=new_frames-self.mean
        return new_frames

    def flow_processing(self, flows):
        # flows = np.reshape(np.asarray(flows), (len(flows), 256, 256, 2)).astype(float)
        flows = np.asarray(flows).astype(float)
        flows = ((2 * (flows - flows.min()) / (flows.max() - flows.min())) - 1) #Todo:源代码应该是整个视频而不是一段视频进行norm
        flows = self.flow_transforms(flows)
        return flows

    def __getitem__(self, i):
        # output format [N,C,T,H,W], [N,2]
        # databegin = time.time()
        key = self.selected_keys[i]
        scores = self.pseudo_labels[key + '.npy']

        if self.type != 'Normal':    # 载入真实标签
            scores = self.annotation_dict[key][0]
            scores = scores[ : len(scores) - len(scores) % 16]
            scores = scores.reshape((-1, 16))
            # scores = np.mean(scores, 1)
            scores = np.max(scores, 1)


        vid_len=self.selected_dict[key]

        if not self.continuous_sampling:
            chosens = random_perturb(vid_len-1, self.clip_num)
        else:
            chosens= np.random.randint(0,vid_len-1-self.clip_num)+np.arange(0, self.clip_num)
        labels = []
        rgb_clips = []
        flow_clips = []

        with h5py.File(self.rgb_h5_file, 'r') as rgb_h5, h5py.File(self.flow_h5_file, 'r') as flow_h5 :
            for chosen in chosens:
                frames = []
                flows = []
                begin=np.random.randint(0,self.dataset_len*2-self.segment_len)
                for j in range(2):
                    frames.extend(rgb_h5[key+'-{0:06d}'.format(chosen+j)][:])
                    flows.extend(flow_h5[key+'-{0:06d}'.format(chosen+j)][:])
                # processbegin = time.time()

                frames=frames[begin:begin+self.segment_len]   # tensor[16,224,224,3]
                frames = self.frame_processing(frames)     #[10*tensor[3,16,224,224]

                flows = flows[begin:begin + self.segment_len]
                flows = self.flow_processing(flows)
                # processend = time.time()
                # print('process take time {}'.format(processend - processbegin))
                if self.ten_crop:
                    frames = torch.stack(frames)

                rgb_clips.append(frames)
                flow_clips.append(flows)


                if chosen >= scores.shape[0]:
                    score=scores[-1]

                else:
                    score_1 = scores[chosen*self.dataset_len // self.score_segment_len]
                    if chosen *self.dataset_len// self.segment_len + 1 < scores.shape[0]:
                        score_2 = scores[chosen*self.dataset_len // self.score_segment_len + 1]
                        # score = max(score_1, score_2)
                        percentage=begin/(self.dataset_len*2-self.segment_len)
                        score=percentage*score_1+(1-percentage)*score_2   # 这里的比例应该写反了
                    else:
                        score = score_1

                if not self.hard_label:
                    if self.type!='Normal':
                        label = np.array([1 - score, score]).astype(np.float32)

                    else:
                        label=np.array([1.,0.]).astype(np.float32)
                    labels.append(label)

                else:
                    if self.type == 'Normal':
                        label = np.array([1., 0.], dtype=np.float32)
                    else:
                        label = np.array([0., 1.], dtype=np.float32)
                    labels.append(label)

        rgb_clips=torch.stack(rgb_clips)
        flow_clips=torch.stack(flow_clips)
        # dataend = time.time()
        # print('take time {}'.format(dataend-databegin))

        return rgb_clips, flow_clips, np.array(labels)


