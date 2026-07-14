import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset

torchvision.disable_beta_transforms_warning()
import warnings
import random

warnings.filterwarnings("ignore")


class LAVIBDataset(Dataset):
    def __init__(self, split_name, data_dir="datasets/LAVIB", height=256, width=256, dur_list=(3, 5, 7),
                 dur_weights=(1.0, 0.0, 0.0)):
        self.dataset_name = split_name
        self.h = height
        self.w = width
        self.dur_list = dur_list
        self.dur_weights = dur_weights
        self.data_root = data_dir
        self.meta_data = self.read_data()

    def __len__(self):
        return len(self.meta_data)

    def read_data(self):
        df = pd.read_csv(os.path.join(self.data_root, "annotations", f"{self.dataset_name}.csv"))
        if self.h <= 256:
            videos = [os.path.join(self.data_root, "segments_downsampled_256",
                                   f"{int(row['name'])}_shot{int(row['shot'])}_{int(row['tmp_crop'])}_{int(row['vrt_crop'])}_{int(row['hrz_crop'])}")
                      for _, row in df.iterrows()]
        else:
            videos = [os.path.join(self.data_root, "segments",
                                   f"{int(row['name'])}_shot{int(row['shot'])}_{int(row['tmp_crop'])}_{int(row['vrt_crop'])}_{int(row['hrz_crop'])}")
                      for _, row in df.iterrows()]
        videos = sorted(videos)
        vid_list = []
        for vid in videos:
            if "train" in self.dataset_name:
                dur = random.choices(self.dur_list, weights=self.dur_weights, k=1)[0]
            else:
                dur = self.dur_list[0]
            for i in range(1, 61 - (dur * 2), dur * 2):
                vid_list.append([vid, (i, i + dur * 2)])
        return vid_list

    @staticmethod
    def crop(ims, h, w):
        _, _, ih, iw = ims.shape
        x = random.randint(0, ih - h)
        y = random.randint(0, iw - w)
        ims = ims[:, :, x:x + h, y:y + w]
        return ims

    def get_frames_from_video(self, vid_path):
        video_fr = torchvision.io.read_video(f"{vid_path[0]}/vid.mp4")[0]
        while video_fr.shape[1] < self.h or video_fr.shape[2] < self.w:
            vid_path = self.meta_data[random.randint(0, len(self.meta_data) - 1)]
            video_fr = torchvision.io.read_video(f"{vid_path[0]}/vid.mp4")[0]
        video_frames = [video_fr[i] for i in range(vid_path[1][0], vid_path[1][1], 2)]
        return torch.stack(video_frames).float().permute(0, 3, 1, 2)

    def get_frames(self, vid_path):
        video_frames = [torchvision.io.read_image(f"{vid_path[0]}/frames/{i:03d}.jpg") for i in range(vid_path[1][0], vid_path[1][1], 2)]
        return torch.stack(video_frames).float()

    def __getitem__(self, index):
        vid_path = self.meta_data[index]
        if self.h <= 256:
            video_frames = self.get_frames_from_video(vid_path)
        else:
            video_frames = self.get_frames(vid_path)
        frames_num = len(video_frames)
        mid_frame_index = (frames_num - 1) // 2
        frame0 = video_frames[0, ...]
        frame1 = video_frames[-1, ...]
        gt = video_frames[mid_frame_index, ...]
        frames = torch.stack((frame0, frame1, gt), dim=0)
        return frames
