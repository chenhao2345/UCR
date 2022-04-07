from __future__ import absolute_import
import warnings

from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17
from .cuhksysu import CUHKSYSU
from .cuhk03 import CUHK03, CUHK03NP
from .ilids import iLIDS
from .viper import VIPeR
from .prid import PRID
from .grid import GRID
from .cuhk01 import CUHK01
from .cuhk02 import CUHK02
from .sensereid import SenseReID
from .ThreeDPeS import ThreeDPeS
from .personx import PersonX

__factory = {
    'market1501': Market1501,
    'dukemtmc-reid': DukeMTMC,
    'msmt17': MSMT17,
    'cuhk-sysu': CUHKSYSU,
    'cuhk03': CUHK03,
    'cuhk03-np': CUHK03NP,
    'ilids': iLIDS,
    'viper': VIPeR,
    'prid2011': PRID,
    'grid': GRID,
    'cuhk01': CUHK01,
    'cuhk02': CUHK02,
    'sensereid': SenseReID,
    '3dpes': ThreeDPeS,
    'personx': PersonX,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. 
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
