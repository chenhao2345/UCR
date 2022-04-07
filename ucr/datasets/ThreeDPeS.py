from __future__ import division, print_function, absolute_import
import glob
import numpy as np
import os.path as osp
import shutil
from collections import defaultdict

from ..utils.serialization import read_json, write_json, mkdir_if_missing
from ..utils.data import BaseImageDataset


class ThreeDPeS(BaseImageDataset):
    """3DPeS
    """
    dataset_dir = ''

    def __init__(self, root='', relabel=True, combineall=False, split_id=0, verbose=True, **kwargs):
        super(ThreeDPeS, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.relabel = relabel
        self.rgb_dir = osp.join(self.dataset_dir, '3DPeS', 'RGB')
        self.meta_path = osp.join(self.dataset_dir, 'meta.json')

        required_files = [self.rgb_dir]
        self.check_before_run(required_files)

        self.prepare_split()
        meta = read_json(self.meta_path)
        train, query, gallery = self.process_split(meta)

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print("=> 3DPeS loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def prepare_split(self):
        if not osp.exists(osp.join(self.meta_path)):
            mkdir_if_missing(osp.join(self.dataset_dir, 'cam_0'))
            mkdir_if_missing(osp.join(self.dataset_dir, 'cam_1'))
            # Collect the person_id and view_id into dict
            images = glob.glob(osp.join(self.rgb_dir, '*.bmp'))
            pdict = defaultdict(lambda: defaultdict(list))
            for imname in images:
                pid, vid = osp.basename(imname).split('_')[0:2]
                pdict[pid][vid].append(imname)
            # Randomly choose half of the views as cam_0, others as cam_1
            identities = []
            for i, pid in enumerate(pdict):
                vids = list(pdict[pid].keys())
                num_views = len(vids)
                np.random.shuffle(vids)
                p_images = [[], []]
                for vid in vids[:(num_views // 2)]:
                    for src_file in pdict[pid][vid]:
                        tgt_file = 'cam_0/{:05d}_{:05d}.bmp'.format(i, len(p_images[0]))
                        shutil.copy(src_file, osp.join(self.dataset_dir, tgt_file))
                        p_images[0].append(tgt_file)
                for vid in vids[(num_views // 2):]:
                    for src_file in pdict[pid][vid]:
                        tgt_file = 'cam_1/{:05d}_{:05d}.bmp'.format(i, len(p_images[1]))
                        shutil.copy(src_file, osp.join(self.dataset_dir, tgt_file))
                        p_images[1].append(tgt_file)
                identities.append(p_images)
            # Save meta information into a json file
            meta = {'name': '3DPeS', 'shot': 'multiple', 'num_cameras': 2}
            meta['identities'] = identities
            write_json(meta, osp.join(self.dataset_dir, 'meta.json'))

    def process_split(self, meta):
        identities = meta['identities']
        num = len(identities)
        pids = np.random.permutation(num)
        train_pids = sorted(pids[num//2:])
        test_pids = sorted(pids[:num//2])

        cam_a_imgs = sorted(glob.glob(osp.join(self.dataset_dir, 'cam_0', '*.bmp')))
        cam_b_imgs = sorted(glob.glob(osp.join(self.dataset_dir, 'cam_1', '*.bmp')))

        train = []
        query = []
        gallery = []

        for img in cam_a_imgs:
            pid = int(osp.basename(img).split('_')[0])
            if pid in test_pids:
                query.append((img, pid, 1))
            else:
                train.append((img, pid, 1))

        for img in cam_b_imgs:
            pid = int(osp.basename(img).split('_')[0])
            if pid in test_pids:
                gallery.append((img, pid, 2))
            else:
                train.append((img, pid, 2))
        return train, query, gallery



