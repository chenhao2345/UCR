from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        img2 = img.copy()

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img2)
        else:
            raise NotImplementedError

        return img1, img2, fname, pid, camid, index


class Preprocessor_index(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False, index=False, transform2=None):
        super(Preprocessor_index, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform2 = transform2
        self.mutual = mutual
        self.index = index

        # self.use_gan=use_gan
        # self.num_cam = num_cam

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index

    def _get_mutual_item(self, index):
        if self.index:
            fname, pid, camid, _ = self.dataset[index]
        else:
            fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        # camstyle_root = osp.dirname(fpath)+'_camstyle'
        # if self.use_gan:
        #     sel_cam = torch.randperm(self.num_cam)[0]
        #     if sel_cam == camid:
        #         if self.root is not None:
        #             fpath = osp.join(self.root, fname)
        #         img = Image.open(fpath).convert('RGB')
        #     else:
        #         # if 'msmt' in self.root:
        #         #     fname = fname[:-4] + '_fake_' + str(sel_cam.numpy() + 1) + '.jpg'
        #         # else:
        #         fname = osp.basename(fname)
        #         fname = fname[:-4] + '_fake_' + str(camid + 1) + 'to' + str(sel_cam.numpy() + 1) + '.jpg'
        #         fpath = osp.join(camstyle_root, fname)
        #         img = Image.open(fpath).convert('RGB')
        # else:
        #     if self.root is not None:
        #         fpath = osp.join(self.root, fname)
        #     img = Image.open(fpath).convert('RGB')

        img = Image.open(fpath).convert('RGB')
        img_mutual = img.copy()
        img2 = img.copy()

        if self.transform is not None:
            img1 = self.transform(img)
            img_mutual = self.transform(img_mutual)
            if self.transform2 is not None:
                img2 = self.transform2(img2)
            else:
                img2 = self.transform(img2)
        else:
            raise NotImplementedError

        return img1, img2, img_mutual, pid, camid