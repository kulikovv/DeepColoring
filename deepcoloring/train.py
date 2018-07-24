import sys

import numpy
import torch

from data import Reader
from halo_loss import halo_loss
from utils import rgba2rgb, normalize, postprocess, symmetric_best_dice, get_as_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(generator, model, mask_builder,
          niter=4000,
          lr=1e-3,
          caption="model",
          k_neg=7,
          verbose=True):

    def print_percent(percent):
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * percent, 5 * percent))
        sys.stdout.flush()

    model.train()

    errors = []

    percent_old = 0
    if verbose:
        print_percent(percent_old)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(niter):
        x, y = generator()
        x_t = torch.from_numpy(x).to(device)

        optimizer.zero_grad()

        pred = model(x_t)
        labels, halo, objects = mask_builder(y)
        loss = halo_loss(pred, labels, halo, objects, k_neg=k_neg)
        errors.append(loss.item())
        loss.backward()
        optimizer.step()

        if 0 == i % 100:
            torch.save(model.state_dict(), caption + '.t7')

        if verbose:
            percent = int(float(i) / float(niter) * 20.)
            if percent_old != percent:
                percent_old = percent
                print_percent(percent)

    if verbose:
        print_percent(100)

    torch.save(model.state_dict(), caption + '.t7')
    numpy.savetxt(caption + '.txt', errors, "%5.5f")
    return model, errors


def evaluate(data, model, min_points):
    """
    Estimates the 
    :param data: 
    :param model: 
    :param min_points: 
    :return: 
    """
    assert isinstance(data, Reader)

    res = []
    model.eval()

    for i in range(len(data)):
        x, y = data[i]
        x = rgba2rgb()(x, True)
        x = normalize(0.5, 0.5, )(x, True)
        x = x.transpose(2, 0, 1)[:, :248, :248]

        vx = torch.from_numpy(numpy.expand_dims(x, 0)).to(device)
        p = model(vx)
        p_numpy = p.detach().cpu().numpy()[0]

        ground_truth = get_as_list(y[:248, :248])
        instances = postprocess(p_numpy, min_points)
        detected_masks = get_as_list(instances)
        res.append(symmetric_best_dice(ground_truth, detected_masks))

    return numpy.array(res).mean(axis=0), numpy.array(res).std(axis=0)
