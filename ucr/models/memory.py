import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd


class Memory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.07, momentum=0.2, K=1024):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.K = K

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, f, f_weak, indexes):
        # inputs: B*2048, features: L*2048
        batchSize = f.size(0)
        k = self.features.clone()
        labels = self.labels
        batch_labels = labels[indexes]
        # print(batch_labels)
        mat = torch.matmul(f, k.transpose(0, 1))
        positives_wogan = []
        negatives_wogan = []
        for i, batch_label in enumerate(batch_labels):
            pos_labels = (labels == batch_label)
            pos = mat[i, pos_labels]
            # perm = torch.randperm(pos.size(0))
            # idx = perm[:1]
            positives_wogan.append(torch.topk(pos, 1, largest=False)[0])
            # positives_wogan.append(pos[idx])

            neg_labels = (labels != batch_label)
            neg = mat[i, neg_labels]
            # perm = torch.randperm(neg.size(0))
            # idx = perm[:self.K]
            negatives_wogan.append(torch.topk(neg, self.K, largest=True)[0])
            # negatives_wogan.append(neg[idx])
        positives_wogan = torch.stack(positives_wogan)
        negatives_wogan = torch.stack(negatives_wogan)
        inter_out_wogan = torch.cat((positives_wogan, negatives_wogan), dim=1) / self.temp

        targets = torch.zeros([batchSize]).cuda().long()
        memory_loss = self.criterion(inter_out_wogan, targets)
        # print(memory_loss)

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.features, 0, indexes.view(-1))
            weight_pos.mul_(self.momentum)
            weight_pos.add_(torch.mul(f_weak, 1 - self.momentum))
            updated_weight = F.normalize(weight_pos)
            self.features.index_copy_(0, indexes, updated_weight)

        return memory_loss
