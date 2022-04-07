from __future__ import print_function, absolute_import

import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
import copy

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from ucr.utils.logging import Logger
from ucr import datasets
from ucr import models
from ucr.evaluators import Evaluator, extract_features
from ucr.utils.data import IterLoader
from ucr.utils.data import transforms as T
from ucr.utils.data.sampler import MoreCameraSampler, RandomIdentitySampler
from ucr.utils.data.preprocessor import Preprocessor_index, Preprocessor
from ucr.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)

    model_1.cuda()
    model_1 = nn.DataParallel(model_1)

    if args.init != '':
        initial_weights = load_checkpoint(args.init)
        copy_state_dict(initial_weights['state_dict'], model_1)

    return model_1


def evaluate_all(args, datasets, evaluator_1_ema):
    rank1 = []
    rank5 = []
    rank10 = []
    mAP = []
    for dataset in datasets:
        test_loader = get_test_loader(dataset, args.height, args.width, 256, args.workers)
        cmc, mAP_1 = evaluator_1_ema.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
        rank1.append(cmc[0])
        rank5.append(cmc[4])
        rank10.append(cmc[9])
        mAP.append(mAP_1)
    print('Average:')
    print('mAP:', sum(mAP) / len(mAP) * 100)
    print('rank1:', sum(rank1) / len(rank1)*100)
    print('rank5:', sum(rank5) / len(rank5)*100)
    print('rank10:', sum(rank10) / len(rank10)*100)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset_market = get_data('market1501', args.data_dir)
    dataset_cuhksysu = get_data('cuhk-sysu', args.data_dir)
    dataset_msmt = get_data('msmt17', args.data_dir)
    dataset_cuhk03 = get_data('cuhk03', args.data_dir)

    dataset_ilids = get_data('ilids', args.data_dir)
    dataset_viper = get_data('viper', args.data_dir)
    dataset_prid2011 = get_data('prid2011', args.data_dir)
    dataset_grid = get_data('grid', args.data_dir)
    dataset_cuhk01 = get_data('cuhk01', args.data_dir)
    dataset_cuhk02 = get_data('cuhk02', args.data_dir)
    dataset_sensereid = get_data('sensereid', args.data_dir)
    dataset_3dpes = get_data('3dpes', args.data_dir)

    datasets = [dataset_market, dataset_cuhksysu, dataset_msmt]
    datasets_unseen = [dataset_viper, dataset_prid2011, dataset_grid, dataset_ilids, dataset_cuhk01, dataset_cuhk02, dataset_sensereid, dataset_cuhk03, dataset_3dpes]
    # Create model
    model_1 = create_model(args)

    # Evaluator
    evaluator_1 = Evaluator(model_1)
    # evaluate_all(args, datasets, evaluator_1)
    evaluate_all(args, datasets_unseen, evaluator_1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Moco Training")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters")
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--tau-c', type=float, default=0.5)
    parser.add_argument('--tau-v', type=float, default=0.09)
    parser.add_argument('--scale-kl', type=float, default=2.0)
    parser.add_argument('--lambda-kl', type=float, default=20.0)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[],
                        help='milestones for the learning rate decay')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    # cluster
    parser.add_argument('--rho', type=float, default=2.2e-3,
                        help="rho percentage, default: 2.2e-3")
    parser.add_argument('--k1', type=int, default=30,
                        help="k1, default: 30")
    parser.add_argument('--min-samples', type=int, default=4,
                        help="min sample, default: 4")
    parser.add_argument('--mem-samples', type=int, default=4,
                        help="mem samples per person, default: 4")
    # init
    parser.add_argument('--init', type=str,
                        default='logs/step2.pth.tar',
                        metavar='PATH')
    end = time.time()
    main()
    print('Time used: {}'.format(time.time()-end))
