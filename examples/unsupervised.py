from __future__ import print_function, absolute_import

import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
import copy

from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from ucr.utils.logging import Logger
from ucr import datasets
from ucr import models
from ucr.trainers import ImageTrainer
from ucr.evaluators import Evaluator, extract_features
from ucr.utils.data import IterLoader
from ucr.utils.data import transforms as T
from ucr.utils.data.sampler import RandomIdentitySampler
from ucr.utils.data.preprocessor import Preprocessor_index, Preprocessor
from ucr.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from ucr.utils.faiss_rerank import compute_jaccard_distance
from ucr.utils.lr_scheduler import WarmupMultiStepLR
from sklearn.metrics import normalized_mutual_info_score
from operator import itemgetter, attrgetter

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, mutual=False, index=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.GaussianBlur([.1, 2.])], p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
    ])

    weak_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomIdentitySampler(train_set, num_instances, video=True)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor_index(train_set, root=dataset.images_dir, transform=train_transformer, mutual=mutual, index=index, transform2=weak_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

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

    model_1_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)

    model_1.cuda()
    model_1_ema.cuda()
    model_1 = nn.DataParallel(model_1)
    model_1_ema = nn.DataParallel(model_1_ema)

    if args.init != '':
        initial_weights = load_checkpoint(args.init)
        copy_state_dict(initial_weights['state_dict'], model_1)
        copy_state_dict(initial_weights['state_dict'], model_1_ema)

    return model_1, model_1_ema


def lifelong_unsupervised_trainer(args, dataset_target, model_1, model_1_ema, optimizer, lr_scheduler, evaluator_1_ema,
                                  model_1_old=None, centers_old=None, dictionary_old=None, id_cam_centers_old=None):
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, 256, args.workers)
    labels_lastepoch = []

    for epoch in range(args.epochs):

        cluster_loader = get_test_loader(dataset_target, args.height, args.width, 256, args.workers,
                                         testset=dataset_target.train)
        dict_f1, _ = extract_features(model_1_ema, cluster_loader, print_freq=50)
        cf = torch.stack(list(dict_f1.values()))

        rerank_dist = compute_jaccard_distance(cf, k1=args.k1, k2=6)
        eps = args.rho
        print('eps in cluster: {:.3f}'.format(eps))
        print('Clustering and labeling...')
        cluster = DBSCAN(eps=eps, min_samples=args.min_samples, metric='precomputed', n_jobs=-1)
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)

        centers = []
        for id in range(num_ids):
            centers.append(torch.mean(cf[labels == id], dim=0))
        centers = torch.stack(centers, dim=0)

        # change pseudo labels
        pseudo_labeled_dataset = []
        pseudo_outliers = 0
        labels_true = []
        cams = []
        # id_cam_centers = {}
        for i, ((fname, pid, cid), label, feat) in enumerate(zip(dataset_target.train, labels, cf)):
            labels_true.append(pid)
            cams.append(cid)
            if label == -1:
                pseudo_outliers += 1
            else:
                pseudo_labeled_dataset.append((fname, label.item(), cid, feat))
                # print(feat)
        cams = np.asarray(cams)
        num_cams = len(np.unique(cams))

        id_cam_centers = {}
        for id in range(num_ids):
            id_cam_centers[id] = []
            for cam in np.unique(cams):
                mask = np.logical_and(labels == id, cams == cam)
                if any(mask):
                    id_cam_centers[id].append(torch.mean(cf[mask], dim=0))

        pseudo_labeled_dataset_newold = []
        pseudo_labeled_dataset_newold.extend(pseudo_labeled_dataset)
        pseudo_labeled_dataset_old = []
        num_ids_newold = num_ids
        num_img_old = 0
        if (centers_old is not None) and (dictionary_old is not None):
            num_ids_newold += centers_old.size(0)
            centers = torch.cat([centers, centers_old], dim=0)
            for k in id_cam_centers_old.keys():
                id_cam_centers[k+num_ids] = id_cam_centers_old[k]

            for j, per_id in enumerate(dictionary_old.values()):
                for (fname, pid, cid, sim) in per_id:
                    num_img_old += 1
                    pseudo_labeled_dataset_old.append((fname, pid+num_ids, cid+num_cams, sim))
                    pseudo_labeled_dataset_newold.append((fname, pid+num_ids, cid+num_cams, sim))
            print('Epoch {}, old dataset has {} labeled samples of {} ids'.
                  format(epoch, num_img_old, centers_old.size(0)))

        print('Label score:', normalized_mutual_info_score(labels_true=labels_true, labels_pred=labels))
        if epoch > 0:
            print('Label score current/last epoch:',
                  normalized_mutual_info_score(labels_true=labels, labels_pred=labels_lastepoch[-1]))
        labels_lastepoch.append(labels)
        print('Epoch {}, current dataset has {} labeled samples of {} ids and {} unlabeled samples'.
              format(epoch, len(pseudo_labeled_dataset), num_ids, pseudo_outliers))
        print('Totally, epoch {} has {} labeled samples of {} ids'.
              format(epoch, len(pseudo_labeled_dataset_newold), num_ids_newold))
        print('Learning Rate:', optimizer.param_groups[0]['lr'])
        train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                               args.batch_size, args.workers, args.num_instances, args.iters,
                                               trainset=pseudo_labeled_dataset, mutual=True, index=True)
        train_loader_target.new_epoch()

        if (centers_old is not None) and (dictionary_old is not None):
            train_loader_target_old = get_train_loader(dataset_target, args.height, args.width,
                                               args.batch_size, args.workers, 2, args.iters,
                                               trainset=pseudo_labeled_dataset_old, mutual=True, index=True)
            train_loader_target_old.new_epoch()
        else:
            train_loader_target_old = None

        # Trainer
        trainer = ImageTrainer(model_1, model_1_ema, num_cluster=num_ids_newold, alpha=args.alpha, tau_c=args.tau_c,
                               scale_kl=args.scale_kl, model_1_old=model_1_old, lambda_kl=args.lambda_kl)

        trainer.train(epoch, train_loader_target, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_target), centers=centers,
                      id_cam_centers=id_cam_centers, num_ids_new=num_ids,
                      train_loader_target_old=train_loader_target_old)

        lr_scheduler.step()

        if (epoch + 1) % args.eval_step == 0:
            cmc, mAP_1 = evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery,
                                                  cmc_flag=True)

    # # save centers, K samples per class
    # randperm = np.random.permutation(num_ids)
    # randperm = randperm[:500]
    pseudo_labeled_dictionary = {}
    for i, (fname, label, cid, feat) in enumerate(pseudo_labeled_dataset):
        sim = F.cosine_similarity(feat.view(1, -1).cuda(), centers[label].view(1, -1).cuda()).item()
        # if label in randperm:
        if label not in pseudo_labeled_dictionary:
            pseudo_labeled_dictionary[label] = list()
        pseudo_labeled_dictionary[label].append((fname, label, cid, sim))
    if (centers_old is not None) and (dictionary_old is not None):
        for k in dictionary_old.keys():
            pseudo_labeled_dictionary[k + num_ids] = dictionary_old[k]
    for j in pseudo_labeled_dictionary.keys():
        pseudo_labeled_dictionary[j] = sorted(pseudo_labeled_dictionary[j], key=itemgetter(3), reverse=True)[:args.mem_samples]
        # if len(pseudo_labeled_dictionary[j]) > args.mem_samples:
        #     pseudo_labeled_dictionary[j] = random.choices(pseudo_labeled_dictionary[j], k=args.mem_samples)

    return model_1, model_1_ema, centers, pseudo_labeled_dictionary, id_cam_centers


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

    dataset_ilids = get_data('ilids', args.data_dir)
    dataset_viper = get_data('viper', args.data_dir)
    dataset_prid2011 = get_data('prid2011', args.data_dir)
    dataset_grid = get_data('grid', args.data_dir)
    dataset_cuhk01 = get_data('cuhk01', args.data_dir)
    dataset_cuhk02 = get_data('cuhk02', args.data_dir)
    dataset_sensereid = get_data('sensereid', args.data_dir)
    dataset_cuhk03 = get_data('cuhk03', args.data_dir)
    dataset_3dpes = get_data('3dpes', args.data_dir)

    datasets = [dataset_market, dataset_cuhksysu, dataset_msmt]
    datasets_unseen = [dataset_viper, dataset_prid2011, dataset_grid, dataset_ilids, dataset_cuhk01, dataset_cuhk02,
                       dataset_sensereid, dataset_cuhk03, dataset_3dpes]
    # Create model
    model_1, model_1_ema = create_model(args)

    # Optimizer
    params = []
    for key, value in model_1.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)

    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=1, warmup_factor=0.1,
                                     warmup_iters=args.warmup_step)

    # Evaluator
    evaluator_1_ema = Evaluator(model_1_ema)
    model_1, model_1_ema, centers, pseudo_labeled_dictionary, id_cam_centers = lifelong_unsupervised_trainer(args, datasets[0], model_1, model_1_ema, optimizer, lr_scheduler, evaluator_1_ema)
    evaluate_all(args, datasets, evaluator_1_ema)
    evaluate_all(args, datasets_unseen, evaluator_1_ema)
    save_checkpoint({'state_dict': model_1_ema.state_dict()}, False, fpath=osp.join(args.logs_dir, 'step1.pth.tar'))
    model_1_old = copy.deepcopy(model_1_ema)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=1, warmup_factor=0.1,
                                     warmup_iters=args.warmup_step)

    for (k_q, v_q), (k_k, v_k) in zip(model_1_ema.state_dict().items(), model_1.state_dict().items()):
        assert k_k == k_q, "state_dict names are different!"
        v_k.copy_(v_q)

    model_1, model_1_ema, centers, pseudo_labeled_dictionary, id_cam_centers = lifelong_unsupervised_trainer(args, datasets[1], model_1, model_1_ema, optimizer, lr_scheduler, evaluator_1_ema, model_1_old=model_1_old, centers_old=centers, dictionary_old=pseudo_labeled_dictionary, id_cam_centers_old=id_cam_centers)
    evaluate_all(args, datasets, evaluator_1_ema)
    evaluate_all(args, datasets_unseen, evaluator_1_ema)
    save_checkpoint({'state_dict': model_1_ema.state_dict()}, False, fpath=osp.join(args.logs_dir, 'step2.pth.tar'))
    model_1_old = copy.deepcopy(model_1_ema)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=1, warmup_factor=0.1,
                                     warmup_iters=args.warmup_step)
    for (k_q, v_q), (k_k, v_k) in zip(model_1_ema.state_dict().items(), model_1.state_dict().items()):
        assert k_k == k_q, "state_dict names are different!"
        v_k.copy_(v_q)

    model_1, model_1_ema, centers, pseudo_labeled_dictionary, id_cam_centers = lifelong_unsupervised_trainer(args, datasets[2], model_1, model_1_ema, optimizer, lr_scheduler, evaluator_1_ema, model_1_old=model_1_old, centers_old=centers, dictionary_old=pseudo_labeled_dictionary, id_cam_centers_old=id_cam_centers)
    evaluate_all(args, datasets, evaluator_1_ema)
    evaluate_all(args, datasets_unseen, evaluator_1_ema)
    save_checkpoint({'state_dict': model_1_ema.state_dict()}, False, fpath=osp.join(args.logs_dir, 'step3.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UCR Training")
    # data
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
                        default='',
                        metavar='PATH')
    end = time.time()
    main()
    print('Time used: {}'.format(time.time()-end))
