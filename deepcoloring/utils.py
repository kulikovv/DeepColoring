import math

import matplotlib
import matplotlib.pyplot as plt
import numpy

import skimage.exposure
import skimage.filters
import skimage.transform
import skimage.util
import skimage.measure
import skimage.draw
import skimage.morphology

from scipy import ndimage as ndi

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
            resize(x, numpy.multiply(x.shape, axes_scale), mode=mode, cval=0, order=order, preserve_range=True,
                   **kwargs)

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


def random_noise(prob=0.5, gain_random=0.001):
    def f(x, is_data=False):
        if is_data:
            if numpy.random.random() < prob:
                x = skimage.util.random_noise(x, var=abs(numpy.random.randn() * gain_random))
        return x

    return f


def blur(sigma=1., prob=0.5, gain_random=0.1):
    def f(x, is_data=False):
        if is_data:
            if numpy.random.random() < prob:
                x = skimage.filters.gaussian(x, sigma=abs(sigma + gain_random * numpy.random.randn()),preserve_range=True,multichannel=True)
        return x

    return f


def random_contrast(low=0.2, high=0.8, gain_random=0.1):
    def f(x, is_data=False):
        if is_data:
            v_min, v_max = numpy.percentile(x, (low + gain_random * numpy.random.randn(),
                                                high + gain_random * numpy.random.randn()))
            x = skimage.exposure.rescale_intensity(x, in_range=(v_min, v_max))
        return x

    return f


def random_gamma(gamma=0.4, gain=0.9, gain_random=0.1, prob=0.5):
    def f(x, is_data=False):
        if is_data:
            if numpy.random.random() < prob:
                x = skimage.exposure.adjust_gamma(x, gamma=gamma + gain_random * numpy.random.randn(),
                                                  gain=gain + gain_random * numpy.random.randn())
        return x

    return f


def random_transform(max_scale, max_angle=90., max_trans=0., keep_aspect_ratio=True):
    """
    Rescales the image according to the scale ratio.
    :param max_scale: The scalar to rescale the image by.
    :param max_angle: Maximum rotation.
    :param max_trans: Maximum translation.
    :param keep_aspect_ratio: Keep aspect ration of the image
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


def visualize(x_np, y_np, min_point=40, draw_text=True, cmap="Set1"):
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)

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

    instances = postprocess(y_np, min_point)
    picture = numpy.argmax(y_np, 0)
    picture[0 == instances] = 0

    for obj in numpy.unique(instances)[1:]:
        innermask = numpy.zeros_like(instances)
        innermask[instances == obj] = 1
        distance = ndi.distance_transform_edt(innermask)
        r, c = numpy.unravel_index(distance.argmax(), distance.shape)
        if draw_text:
            ax2.text(c - 3, r + 3, r'{}'.format(int(obj)), fontdict=font)

    ax2.imshow(picture, cmap=matplotlib.colors.ListedColormap(color_map[:y_np.shape[0]]))

    y_np = softmax(y_np)
    f2, ax = plt.subplots(int(math.ceil(y_np.shape[0] / 3.)), 3, figsize=(20, 10))
    for index in range(y_np.shape[0]):
        color_index = -1
        color_index2 = index

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('xxx',
                                                                   [color_map[color_index], color_map[color_index2]])
        ax[index / 3, index % 3].imshow(y_np[index, :, :], vmin=0, vmax=1, cmap=cmap)

    return f1, f2


def best_dice(l_a, l_b):
    """
    Best Dice function
    :param l_a: list of binary instances masks
    :param l_b: list of binary instances masks
    :return: best dice estimation
    """
    result = 0
    for a in l_a:
        best_iter = 0
        for b in l_b:
            inter = 2 * float(numpy.sum(a * b)) / float(numpy.sum(a) + numpy.sum(b))
            if inter > best_iter:
                best_iter = inter
        result += best_iter
    if 0 == len(l_a):
        return 0

    return result / len(l_a)


def symmetric_best_dice(l_ar, l_gr):
    """
    Symmetric Best Dice function
    :param l_ar: list of output binary instances masks
    :param l_gr: list of binary ground truth masks
    :return: Symmetric best dice estimation
    """
    return numpy.min([best_dice(l_ar, l_gr), best_dice(l_gr, l_ar)])


def get_as_list(indexes):
    """
    Convert indexes to list
    """
    objects = []
    pixels = numpy.unique(indexes)
    for l, v in enumerate(pixels[1:]):
        bin_mask = numpy.zeros_like(indexes)
        bin_mask[indexes == v] = 1
        objects.append(bin_mask)
    return objects


def postprocess(mapsx, min_point):
    """
    Segment a maps to individual objects
    :param maps: numpy array wxhxd
    :param thresholds: list of threshold of length d-1, applied to probability maps
    :param min_point: list of minimal connected component of length d-1
    :return: int32 image with unique id for each instance
    """

    if not isinstance(min_point, list):
        min_point = [min_point] * (mapsx.shape[2] - 1)

    assert (mapsx.shape[2] == (len(min_point) + 1))

    object_index = 1
    argmaxes = numpy.argmax(mapsx, axis=0)
    output = numpy.zeros_like(mapsx[0, :, :])

    for i in range(1, mapsx.shape[0]):
        contours = skimage.measure.find_contours(argmaxes == i, 0.5)
        for c in contours:
            rr, cc = skimage.draw.polygon(c[:, 0], c[:, 1])
            if len(rr) > min_point[i - 1]:
                output[rr, cc] = object_index
                object_index += 1

    return output