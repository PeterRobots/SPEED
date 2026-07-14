import os

import torch
import torchvision
from torch.utils.data import Dataset
torchvision.disable_beta_transforms_warning()
import warnings
warnings.filterwarnings("ignore")


class XTestDataset(Dataset):
    def __init__(self, data_dir="datasets/X4K1000FPS/test", res="2k"):
        self.data_root = data_dir
        self.res = res
        self.meta_data = self.read_data()

    def __len__(self):
        return len(self.meta_data)

    def read_data(self):
        vid_list = []
        for type_dir in ["Type1", "Type2", "Type3"]:
            for sub_dir in sorted(os.listdir(os.path.join(self.data_root, type_dir))):
                for intFrame in range(0, 32, 32):
                    f0 = f"{self.data_root}/{type_dir}/{sub_dir}/{intFrame:04d}.png"
                    f1 = f"{self.data_root}/{type_dir}/{sub_dir}/{(intFrame + 32):04d}.png"
                    gt = f"{self.data_root}/{type_dir}/{sub_dir}/{(intFrame + 16):04d}.png"
                    triplet = (f0, f1, gt)
                    vid_list.append(triplet)
        return vid_list

    def __getitem__(self, index):
        vid_path = self.meta_data[index]
        frame0 = torchvision.io.read_image(vid_path[0])
        frame1 = torchvision.io.read_image(vid_path[1])
        gt = torchvision.io.read_image(vid_path[2])
        if self.res == "1k":
            frame0 = torch.nn.functional.interpolate(frame0.unsqueeze(0), size=(540, 1024), mode="bilinear").squeeze(0)
            frame1 = torch.nn.functional.interpolate(frame1.unsqueeze(0), size=(540, 1024), mode="bilinear").squeeze(0)
            gt = torch.nn.functional.interpolate(gt.unsqueeze(0), size=(540, 1024), mode="bilinear").squeeze(0)
        elif self.res == "2k":
            frame0 = torch.nn.functional.interpolate(frame0.unsqueeze(0), size=(1080, 2048), mode="bilinear").squeeze(0)
            frame1 = torch.nn.functional.interpolate(frame1.unsqueeze(0), size=(1080, 2048), mode="bilinear").squeeze(0)
            gt = torch.nn.functional.interpolate(gt.unsqueeze(0), size=(1080, 2048), mode="bilinear").squeeze(0)
        # elif self.res == "4k":
        #     frame0 = frame0[:, 540:-540, 1024:-1024]
        #     frame1 = frame1[:, 540:-540, 1024:-1024]
        #     gt = gt[:, 540:-540, 1024:-1024]
        frames = torch.stack((frame0, frame1, gt), dim=0)
        return frames
