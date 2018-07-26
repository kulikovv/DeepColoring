import sys

import numpy
import torch

from halo_loss import halo_loss


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
            percent = int(float(i) / float(niter)*20.)
            if percent_old != percent:
                percent_old = percent
                print_percent(percent)

    if verbose:
        print_percent(20)

    torch.save(model.state_dict(), caption + '.t7')
    numpy.savetxt(caption + '.txt', errors, "%5.5f")
    return model, errors
