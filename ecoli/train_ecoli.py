directory = ""
if __name__ == '__main__':
    from os import sys, path

    directory = path.dirname(path.abspath(__file__))
    sys.path.append(path.dirname(directory))

import os
from os import listdir
from os.path import join

import numpy
import torch

import deepcoloring as dc
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clip_mask_builder(builder, size=256, target=156):
    padding_shift = (size - target) / 2

    def func(y):
        return builder(y[:, padding_shift:padding_shift + target, padding_shift:padding_shift + target])
    return func

class ReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='"Training DeepColoring on Ecoli dataset"')
    # Path to the folder

    parser.add_argument("-s",
                        dest="basepath",
                        action=ReadableDir,
                        help="Path to CVPPP A1 dataset folder",
                        default="/media/hpc-4_Raid/vkulikov/Microscopy/Ecoli/ecoli_plus")

    args = parser.parse_args()

    basepath = args.basepath

    rgb = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_rgb.png')])
    labels = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_label.png')])
    # Check the names are paired correctly
    assert numpy.array([img[:-7] == lbl[:-9] for img, lbl in zip(rgb, labels)]).all() == True

    numpy.random.seed(1203412412)
    indexes = numpy.random.permutation(len(rgb))
    perm_rgb = numpy.array(rgb)[indexes].tolist()
    perm_labels = numpy.array(labels)[indexes].tolist()

    train_data = dc.Reader(perm_rgb[:-10], perm_labels[:-10], 1)
    valid_data = dc.Reader(perm_rgb[-10:], perm_labels[-10:], 1)

    transforms = [dc.clip_patch((256, 256)),
                  dc.flip_horizontally(),
                  dc.flip_vertically(),
                  dc.rotate90(),
                  dc.rgb2gray(),
                  dc.normalize(0.5, 0.5)]

    generator = train_data.create_batch_generator(30, transforms=transforms)
    mask_builder = clip_mask_builder(dc.build_halo_mask(fixed_depth=100,margin=35,min_fragment=10))

    net = dc.EUnet(1, 6, k=2, s=4, depth=3, l=1, padding=0, use_dropout=True).to(device)
    model, errors = dc.train(generator=generator,
                             model=net,
                             mask_builder=mask_builder,
                             niter=7000,
                             k_neg=5.,
                             lr=1e-3,
                             caption=join(directory, "model"))
