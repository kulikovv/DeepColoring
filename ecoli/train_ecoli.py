if __name__ == '__main__':
    from os import sys, path

    directory = path.dirname(path.dirname(path.abspath(__file__)))
    sys.path.append(directory)

from os import listdir
from os.path import join

import numpy
import torch

import deepcoloring as dc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clip_mask_builder(builder, size=256, target=132):
    padding_shift = (size - target) / 2

    def func(y):
        return builder(y[:, padding_shift:padding_shift + target, padding_shift:padding_shift + target])

    return func


if __name__ == "__main__":
    print("Training on Ecoli dataset")
    basepath = "/media/Raid/vkulikov/Microscopy/Ecoli/ecoli_plus"

    rgb = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_rgb.png')])
    labels = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_label.png')])
    # Check the names are paired correctly
    assert numpy.array([img[:-7] == lbl[:-9] for img, lbl in zip(rgb, labels)]).all() == True

    numpy.random.seed(1203412412)
    indexes = numpy.random.permutation(len(rgb))
    perm_rgb = numpy.array(rgb)[indexes].tolist()
    perm_labels = numpy.array(labels)[indexes].tolist()

    train_data = dc.Reader(perm_rgb[:-10], perm_labels[:-10], 2)
    valid_data = dc.Reader(perm_rgb[-10:], perm_labels[-10:], 2)

    transforms = [dc.clip_patch((256, 256)),
                  dc.flip_horizontally(),
                  dc.flip_vertically(),
                  dc.rotate90(),
                  dc.blur(),
                  dc.normalize(0.5, 0.5)]

    generator = train_data.create_batch_generator(20, transforms=transforms)
    mask_builder = clip_mask_builder(dc.build_halo_mask(fixed_depth=100))

    net = dc.EUnet(3, 6, 4, 1, 1, depth=4, padding=0, use_dropout=True).to(device)
    model, errors = dc.train(generator=generator, model=net, mask_builder=mask_builder, niter=10000)
