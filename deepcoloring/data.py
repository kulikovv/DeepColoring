import random

import numpy as np
import skimage.transform as transform
from skimage.io import imread


class Reader:
    def __init__(self, image_list, labels_list=[], scale=1):
        assert isinstance(scale, int)
        if len(labels_list) is not 0:
            assert len(image_list) == len(labels_list)

        self.scale = scale

        self.images = image_list
        self.labels = labels_list

        self.train_probabilities = np.ones(len(image_list)) / float(len(image_list))

        self.cache = {}

    def __getitem__(self, idx):

        if idx not in self.cache:
            rgb = imread(self.images[idx], plugin='pil')
            rgb = transform.resize(rgb, np.array(rgb.shape[:2]) / self.scale)

            label = None
            if idx < len(self.labels):
                label = imread(self.labels[idx],True, plugin='pil')
                label = np.digitize(label,bins=np.unique(label))-1
                label = label[::self.scale, ::self.scale]
                label = label[:rgb.shape[0], :rgb.shape[1]]

            self.cache[idx] = (rgb, label)

        return self.cache[idx]

    def create_batch_generator(self, batch_size=3, transforms=[]):
        assert len(self.images) == len(self.labels)

        def f():
            images = []
            labels = []

            for t in transforms:
                if hasattr(t, 'prepare'):
                    t.prepare()

            for index in range(batch_size):
                idx = np.random.choice(range(len(self.images)), 1, p=self.train_probabilities)[0]
                i, l = self[idx]

                for t in transforms:
                    seed = random.randint(0, 2 ** 32 - 1)
                    np.random.seed(seed)
                    i = t(i, True)
                    np.random.seed(seed)
                    l = t(l)

                images.append(np.expand_dims(i.transpose(2, 0, 1), 0))
                labels.append(np.expand_dims(l, 0))

            labels = np.vstack(labels)

            return np.vstack(images), labels

        return f
