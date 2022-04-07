from __future__ import division, print_function, absolute_import
import copy
import glob
import os.path as osp
from ..utils.serialization import read_json, write_json, mkdir_if_missing
from ..utils.data import BaseImageDataset


class SenseReID(BaseImageDataset):
    """SenseReID.

    This dataset is used for test purpose only.

    Reference:
        Zhao et al. Spindle Net: Person Re-identification with Human Body
        Region Guided Feature Decomposition and Fusion. CVPR 2017.

    URL: `<https://drive.google.com/file/d/0B56OfSrVI8hubVJLTzkwV2VaOWM/view>`_

    Dataset statistics:
        - query: 522 ids, 1040 images.
        - gallery: 1717 ids, 3388 images.
    """
    dataset_dir = ''
    dataset_url = None

    def __init__(self, root='', verbose=True, **kwargs):
        super(SenseReID, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.query_dir = osp.join(self.dataset_dir, 'SenseReID', 'test_probe')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'SenseReID', 'test_gallery'
        )

        required_files = [self.dataset_dir, self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)

        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        # relabel
        g_pids = set()
        for _, pid, _ in gallery:
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        query = [
            (img_path, pid2label[pid], camid)
            for img_path, pid, camid in query
        ]
        gallery = [
            (img_path, pid2label[pid], camid)
            for img_path, pid, camid in gallery
        ]
        train = copy.deepcopy(query) + copy.deepcopy(gallery) # dummy variable
        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print("=> SenseReID loaded")
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

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        data = []

        for img_path in img_paths:
            img_name = osp.splitext(osp.basename(img_path))[0]
            pid, camid = img_name.split('_')
            pid, camid = int(pid), int(camid)
            data.append((img_path, pid, camid))

        return data
