# UCR

Implementation of paper ["Unsupervised Lifelong Person Re-identification via Contrastive Rehearsal"](https://arxiv.org/pdf/2203.06468.pdf).

## Installation

```shell
conda create -n env_ucr python=3.6
source activate env_ucr 
pip install numpy torch==1.4.0 torchvision==0.5.0 h5py six Pillow scipy sklearn metric-learn tqdm faiss-gpu==1.6.3
python setup.py develop
```

## Prepare Datasets

```shell
cd examples && mkdir data
```
Download the raw datasets [Market-1501](http://www.liangzheng.org/Project/project_reid.html), 
[Cuhk-Sysu](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html), 
[MSMT17](http://www.pkuvmc.com/publications/msmt17.html), 
[VIPeR](http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip), 
[PRID2011](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/), 
[GRID](http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip), 
[iLIDS](http://www.eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz), 
[CUHK01](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html), 
[CUHK02](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html), 
[SenseReID](https://drive.google.com/file/d/0B56OfSrVI8hubVJLTzkwV2VaOWM/view), 
[CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) and
[3DPeS](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=16), 
and then unzip them under the directory like
```
UCR/examples/data
├── market1501
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   └── query/
├── cuhk-sysu
│   └── CUHK-SYSU
│       ├── Image/
│       └── annotation/
├── msmt17
│   └── MSMT17_V2
├── viper
│   └── VIPeR
├── prid2011
│   └── prid_2011
├── grid
│   └── underground_reid
├── ilids
│   └── i-LIDS_Pedestrian
├── cuhk01
│   └── campus
├── cuhk02
│   └── Dataset
├── sensereid
│   └── SenseReID
├── cuhk03
│   └── cuhk03_release
└── 3dpes
    └── 3DPeS
```


## Train:
Train UCR on default order (Market to Cuhk-Sysu to MSMT17). 
The results reported in the paper were obtained with **4 GPUs**.
#### Unsupervised lifelong training
```shell
sh unsupervised_lifelong.sh
```

#### Supervised lifelong training
```shell
sh supervised_lifelong.sh
```

## Test:
```shell
python examples/test.py --init examples/logs/step3.pth.tar
```

## Citation
If you find this project useful, please kindly star our project and cite our paper.
```bibtex
@article{chen2022unsupervised,
  title={Unsupervised Lifelong Person Re-identification via Contrastive Rehearsal},
  author={Chen, Hao and Lagadec, Benoit and Bremond, Francois},
  journal={arXiv preprint arXiv:2203.06468},
  year={2022}
}

```
