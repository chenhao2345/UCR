from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F
from ucr.loss import CrossEntropyLabelSmooth
from .utils.meters import AverageMeter
from .evaluation_metrics import accuracy


class ImageTrainer(object):
    def __init__(self, model_1, model_1_ema, num_cluster=500, alpha=0.999, tau_c=0.5, scale_kl=2.0, model_1_old=None, lambda_kl=20):
        super(ImageTrainer, self).__init__()
        self.model_1 = model_1
        self.model_1_ema = model_1_ema
        self.model_1_old = model_1_old
        self.alpha = alpha

        self.tau_c = tau_c
        self.scale_kl = scale_kl
        self.lambda_kl = lambda_kl

        self.celoss = CrossEntropyLabelSmooth(num_cluster)
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.crosscam_epoch = 0
        self.beta = 0.07
        self.bg_knn = 50

    def train(self, epoch, data_loader_target, optimizer, print_freq=1, train_iters=200,
              centers=None, id_cam_centers=None, num_ids_new=None, train_loader_target_old=None):
        self.model_1.train()
        self.model_1_ema.train()
        centers = centers.cuda()
        centers_new = centers[:num_ids_new]
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ccl = AverageMeter()
        losses_ccl_old = AverageMeter()
        losses_cam = AverageMeter()
        losses_kl_old = AverageMeter()
        precisions = AverageMeter()

        percam_tempV = []
        concate_intra_class = []
        for key in id_cam_centers.keys():
            percam_tempV.extend(id_cam_centers[key])
            concate_intra_class.append(torch.tensor([key] * len(id_cam_centers[key])).cuda())
        percam_tempV = torch.stack(percam_tempV, dim=0).cuda()
        concate_intra_class = torch.cat(concate_intra_class, dim=0).cuda()

        end = time.time()
        for i in range(train_iters):
            # print('Learning Rate:', optimizer.param_groups[0]['lr'])
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)
            # process inputs
            inputs_1, inputs_weak, targets, inputs_2, cids = self._parse_data(target_inputs)
            b, c, h, w = inputs_1.size()

            # ids for ShuffleBN
            shuffle_ids, reverse_ids = self.get_shuffle_ids(b)

            f_out_t1 = self.model_1(inputs_1)
            p_out_t1 = torch.matmul(f_out_t1, centers_new.transpose(1,0))/self.tau_c

            loss_cam = torch.tensor([0.]).cuda()
            for cc in torch.unique(cids):
                inds = torch.nonzero(cids == cc).squeeze(-1)
                percam_targets = targets[inds]
                percam_feat = f_out_t1[inds]

                # # intra-camera loss
                # mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
                # mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
                # # percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], self.alpha)
                # percam_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(self.percam_memory[cc].t()))
                # percam_inputs /= self.beta  # similarity score before softmax
                # loss_cam += F.cross_entropy(percam_inputs, mapped_targets)

                # global loss
                associate_loss = 0
                # target_inputs = percam_feat.mm(percam_tempV.t().clone())
                target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
                temp_sims = target_inputs.detach().clone()
                target_inputs /= self.beta

                for k in range(len(percam_feat)):
                    ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                    temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                    sel_ind = torch.sort(temp_sims[k])[1][-self.bg_knn:]
                    concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                    concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).cuda()
                    concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                    associate_loss += -1 * (
                                F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                            0)).sum()

                loss_cam += 0.5 * associate_loss / len(percam_feat)

            with torch.no_grad():
                inputs_1 = inputs_1[shuffle_ids]
                f_out_t1_ema = self.model_1_ema(inputs_1)
                f_out_t1_ema = f_out_t1_ema[reverse_ids]

                # inputs_2 = inputs_2[shuffle_ids]
                # f_out_t2_ema = self.model_1_ema(inputs_2)
                # f_out_t2_ema = f_out_t2_ema[reverse_ids]

                inputs_weak = inputs_weak[shuffle_ids]
                f_out_weak_ema = self.model_1_ema(inputs_weak)
                f_out_weak_ema = f_out_weak_ema[reverse_ids]

            if self.model_1_old is not None:
                # centers_old = centers[num_ids_new:]
                target_inputs_old = train_loader_target_old.next()
                inputs_1_old, inputs_weak_old, targets_old, inputs_2_old, cids_old = self._parse_data(target_inputs_old)
                f_out_t1_old = self.model_1(inputs_1_old)
                p_out_t1_old = torch.matmul(f_out_t1_old, centers.transpose(1, 0)) / self.tau_c
                loss_ccl_old = self.celoss(p_out_t1_old, targets_old)

                with torch.no_grad():
                    inputs_1_old = inputs_1_old[shuffle_ids]
                    f_out_old_ema = self.model_1_ema(inputs_1_old)
                    f_out_old_ema = f_out_old_ema[reverse_ids]

                    # inputs_2_old = inputs_2_old[shuffle_ids]
                    # f_out_old2_ema_old = self.model_1_old(inputs_2_old)
                    # f_out_old2_ema_old = f_out_old2_ema_old[reverse_ids]

                    inputs_weak_old = inputs_weak_old[shuffle_ids]
                    f_out_old_ema_old = self.model_1_old(inputs_weak_old)
                    f_out_old_ema_old = f_out_old_ema_old[reverse_ids]

                loss_ccl = self.celoss(p_out_t1, targets)
                loss_kl_old = self.kl(F.softmax(torch.matmul(F.normalize(f_out_t1_old), F.normalize(f_out_old_ema).transpose(1,0))/self.scale_kl, dim=1).log(),
                                  F.softmax(torch.matmul(F.normalize(f_out_old_ema_old), F.normalize(f_out_old_ema_old).transpose(1,0))/self.scale_kl, dim=1))*self.lambda_kl
                loss = loss_ccl + loss_cam + loss_ccl_old + loss_kl_old
            else:
                loss_ccl = self.celoss(p_out_t1, targets)
                loss_ccl_old = torch.tensor([0.]).cuda()
                loss_kl_old = torch.tensor([0.]).cuda()
                loss = loss_ccl + loss_cam

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch * len(data_loader_target) + i)

            prec_1, = accuracy(p_out_t1.data, targets.data)

            losses_ccl.update(loss_ccl.item())
            losses_ccl_old.update(loss_ccl_old.item())
            losses_cam.update(loss_cam.item())
            losses_kl_old.update(loss_kl_old.item())
            precisions.update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ccl {:.3f}\t'
                      'Loss_ccl_old {:.3f}\t'
                      'Loss_cam {:.3f}\t'
                      'Loss_kl_old {:.3f}\t'
                      'Prec {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ccl.avg,
                              losses_ccl_old.avg,
                              losses_cam.avg,
                              losses_kl_old.avg,
                              precisions.avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, img_mutual, pids, cids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        inputs_mutual = img_mutual.cuda()
        targets = pids.cuda()
        cids = cids.cuda()
        return inputs_1, inputs_2, targets, inputs_mutual, cids

    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for ShuffleBN"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds