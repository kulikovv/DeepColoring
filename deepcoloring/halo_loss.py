import warnings

import numpy
import torch
import torch.nn.functional as F
from skimage.draw import circle


def flatten(x):
    return x.view(x.size(0), -1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_halo_mask(fixed_depth=30, margin=21, min_fragment=10):
    """
    Function builds a configuration for halo region building
    :param fixed_depth: Maximum object on an image
    :param margin: The size of halo region
    :param min_fragment: Minimal size of an object on the image
    :return: a function for generation labels, masks and object_lists used by halo loss
    """
    assert margin % 2 is not 0, "Margin should be odd"

    rr, cc = circle(margin / 2, margin / 2, margin / 2 + 1, shape=(margin, margin))
    structure_element = numpy.zeros((margin, margin))
    structure_element[rr, cc] = 1
    structure_element = numpy.repeat(numpy.expand_dims(numpy.expand_dims(structure_element, 0), 0), fixed_depth, 0)

    sel = torch.from_numpy(structure_element).float().to(device)

    def f(label):
        """
        
        :param label: batch of instance levels each instance must have unique id
        :return:  labels, masks and object_lists used by halo loss
        """
        back = numpy.zeros((label.shape[0], fixed_depth, label.shape[1], label.shape[2]))
        object_list = []
        for i in range(label.shape[0]):
            bincount = numpy.bincount(label[i].flatten())
            pixels = numpy.where(bincount > min_fragment)[0]
            if len(pixels) > fixed_depth:
                pixels = pixels[:fixed_depth]
                warnings.warn("Not all objects fits in fixed depth", RuntimeWarning)

            for l, v in enumerate(pixels):
                back[i, l, label[i] == v] = 1.
            object_list.append(numpy.array(range(l + 1)))

        labels = torch.from_numpy(back).float().to(device)
        masks = F.conv2d(labels, sel, groups=fixed_depth, padding=margin / 2)
        
        masks[masks > 0] = 1.
        masks[labels > 0] = 2.
        masks[:, 0, :, :] = 1.
        
        weights=masks.sum(-1,keepdim=True).sum(-2,keepdim=True)
        weights[weights==0.]=1.
        
        masks = masks/weights
        
        return labels, masks, object_list

    return f


def halo_loss(predicted, labels, weights, obj, k_neg=7.):
    """
    Loss that seeks the best split of instances to several channels
    :param predicted: tensor output from neural network(NxCxWxH)
    :param labels: ground truth binary mask for each object
    :param weights: weights for each object and it's margin
    :param obj: list for each sample indexes of labels, containing any meaningful information
    :param k_neg: negative positioning influance 
    :return: torch variable contains calculation graph
    """

    global_loss = torch.FloatTensor([0]).to(device)
    x_lsm = F.log_softmax(predicted, dim=1)
    x_sm = F.softmax(predicted, dim=1)

    v_weights = weights
    v_labels = labels

    for i in range(predicted.size(0)):  # Foreach sample

        # predicted[0] is preserved for background
        bg = v_weights[i, 0] * v_labels[i, 0]
        global_loss += -torch.sum(bg * x_lsm[i, 0]) / torch.sum(bg)

        if 0 == len(obj[i][1:]):
            continue

        indexes = torch.LongTensor(obj[i][1:]).to(device)  # One is for labels

        valid_labels = v_labels[i].index_select(0, indexes)
        valid_weights = v_weights[i].index_select(0, indexes)

        target_pos = valid_labels * valid_weights
        target_neg = (1. - valid_labels) * valid_weights

        positive = -torch.mm(flatten(x_lsm[i, 1:]),
                             flatten(target_pos).transpose(1, 0)) / torch.sum(target_pos)

        negative = -torch.mm(flatten(torch.log(1. - x_sm[i, 1:] + 1e-12)),
                             flatten(target_neg).transpose(1, 0)) / torch.sum(target_neg)

        values, indices = torch.min(positive + k_neg * negative, 0)
        global_loss += torch.sum(positive[indices, torch.arange(0, positive.size(1)).long().to(device)])

    # Normalize w.r.t. image size
    return global_loss * (1000. / (predicted.size(2) * predicted.size(3)))
