import math

import matplotlib
import matplotlib.pyplot as plt
import numpy
import skimage.transform
from scipy import ndimage as ndi
from skimage import measure
from skimage.draw import polygon
from skimage.morphology import square, closing

"""
Note: Standardization and transforms assumes
that x comes in WxHxC format from the reader
"""


def flip_horizontally(prob=0.5):
    assert 0. < prob < 1.

    def f(x, is_data=False):
        if numpy.random.random() < prob:
            return numpy.fliplr(x)
        return x

    return f


def flip_vertically(prob=0.5):
    assert 0. < prob < 1.

    def f(x, is_data=False):
        if numpy.random.random() < prob:
            return numpy.flipud(x)
        return x

    return f


def rotate90(prob=0.5):
    assert 0. < prob < 1.

    def f(x, is_data=False):
        if numpy.random.random() < prob:
            return numpy.rot90(x, 2, axes=(0, 1))
        return x

    return f


def rescale(scale, **kwargs):
    """
    Rescales the image according to the scale ratio.
    :param scale: The scalar to rescale the image by.
    :param kwargs: Additional arguments for skimage.transform.resize.
    :return: The rescale function.
    """

    axes_scale = (scale, scale, 1.0)

    def f(x, is_data=False):
        mode = 'constant'
        order = 0
        if is_data:
            mode = 'reflect'
            order = 1
        return skimage.transform. \
            resize(x, numpy.multiply(x.shape, axes_scale), mode=mode,cval=0, order=order, preserve_range=True, **kwargs)

    return f


def random_scale(scale_variance=0.2, **kwargs):
    def f(x, is_data=False):
        mode = 'constant'
        order = 0
        if is_data:
            mode = 'reflect'
            order = 1

        s = 1. + numpy.clip(scale_variance * numpy.random.randn(), -scale_variance, scale_variance)
        return skimage.transform. \
            rescale(x, s, mode=mode, order=order, cval=0, preserve_range=True, **kwargs)

    return f


def random_contrast(contrast, clip_value=0.5):
    def f(x, is_data=False):
        if is_data:
            cont = 1. + contrast * numpy.random.randn()
            x = numpy.clip(x * cont, -clip_value, clip_value)
        return x

    return f


def random_brightness(brightness, clip_value=0.5):
    def f(x, is_data=False):
        if is_data:
            x = numpy.clip(x + brightness * numpy.random.randn(), -clip_value, clip_value)
        return x

    return f


def random_transform(max_scale, max_angle, max_trans, keep_aspect_ratio=True):
    """
    Rescales the image according to the scale ratio.
    :param scale: The scalar to rescale the image by.
    :param kwargs: Additional arguments for skimage.transform.resize.
    :return: The rescale function.
    """

    def f(x, is_data=False):

        if keep_aspect_ratio:
            scalex = scaley = 1. + numpy.random.randn() * max_scale
        else:
            scalex = 1. + numpy.random.randn() * max_scale
            scaley = 1. + numpy.random.randn() * max_scale

        shift_y, shift_x = numpy.array(x.shape[:2]) / 2.
        shift = skimage.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        shift_inv = skimage.transform.SimilarityTransform(translation=[shift_x + numpy.random.randn() * max_trans,
                                                                       shift_y + numpy.random.randn() * max_trans])
        trans = skimage.transform.SimilarityTransform(
            rotation=numpy.deg2rad(numpy.random.uniform(-max_angle, max_angle)),
            scale=(scalex, scaley))
        final_transform = (shift + (trans + shift_inv)).inverse

        mode = 'constant'
        order = 0
        if is_data:
            mode = 'reflect'
            order = 1

        return skimage.transform.warp(x, final_transform, mode=mode, order=order, cval=0, preserve_range=True)

    return f


def rotate(max_angle=360):
    def f(x, is_data=False):
        k = numpy.random.uniform(-max_angle, max_angle)
        mode = 'constant'
        order = 0
        if is_data:
            mode = 'reflect'
            order = 1
        return skimage.transform.rotate(x, k, mode=mode, cval=0, order=order, preserve_range=True)

    return f


def rgba2rgb():
    def f(x, is_data=False):
        if is_data:
            x = x[:, :, :3].astype(numpy.float32)
            x /= 255.
        return x

    return f


def normalize(mean, std):
    def f(x, is_data=False):
        if is_data:
            x = (x - mean) / std
        return x

    return f


def vgg_normalize():
    return normalize(numpy.array([0.485, 0.456, 0.406]), numpy.array([0.229, 0.224, 0.225]))


def clip_patch(size):
    assert len(size) == 2

    def f(x, is_data=False):
        cx = numpy.random.randint(0, x.shape[0] - size[0])
        cy = numpy.random.randint(0, x.shape[1] - size[1])
        return x[cx:cx + size[0], cy:cy + size[1]]

    return f


def clip_patch_random(minsize, maxsize):
    assert len(minsize) == 2
    assert len(maxsize) == 2

    def f(x, is_data=False):
        cx = numpy.random.randint(0, x.shape[0] - f.size[0])
        cy = numpy.random.randint(0, x.shape[1] - f.size[1])
        return x[cx:cx + f.size[0], cy:cy + f.size[1]]

    def prepare():
        f.size = (numpy.random.randint(minsize[0], maxsize[0]) * 8, numpy.random.randint(minsize[1], maxsize[1]) * 8)

    f.prepare = prepare
    f.prepare()

    return f


def visualize(x_np, y_np, cmap="Set1", min_point=40, draw_text=True, cl=5):
    font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
        }
    color_map = [(1., 1., 1., 1.)]
    colors = matplotlib.cm.get_cmap(cmap)
    for index in range(y_np.shape[2]):
        color_map.append(colors(index))
    color_map.append((0., 0., 0., 1.))

    f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('source')
    ax1.imshow(x_np, cmap='gray')
    ax2.set_title('result')
    picture = numpy.argmax(y_np, axis=2)

    object_index = 1
    for i in range(1, numpy.max(picture) + 1):
        mask = numpy.zeros_like(picture)
        mask[picture == i] = 1

        mask = closing(mask, square(cl))

        contours = measure.find_contours(mask, 0.99)
        for c in contours:
            rr, cc = polygon(c[:, 0], c[:, 1])

            if len(rr) > min_point:
                innermask = numpy.zeros_like(picture)
                innermask[rr, cc] = 1
                distance = ndi.distance_transform_edt(innermask)
                r, c = numpy.unravel_index(distance.argmax(), distance.shape)
                if draw_text:
                    ax2.text(c - 3, r + 3, r'{}'.format(object_index), fontdict=font)
                object_index += 1
            else:
                picture[rr, cc] = 0
    ax2.imshow(picture, cmap=matplotlib.colors.ListedColormap(color_map[:y_np.shape[2]]))

    f2, ax = plt.subplots(int(math.ceil(y_np.shape[2] / 3.)), 3, figsize=(20, 10))
    for index in range(y_np.shape[2]):
        color_index = -1
        color_index2 = index

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('xxx',
                                                                   [color_map[color_index], color_map[color_index2]])
        ax[index / 3, index % 3].imshow(y_np[:, :, index], vmin=0, vmax=1, cmap=cmap)

    return f1, f2
