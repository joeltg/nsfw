import settings
import caffe
import numpy as np

import numpy as np
import math, random
import sys, subprocess
from IPython.display import clear_output, Image, display
from scipy.misc import imresize
from numpy.linalg import norm
from numpy.testing import assert_array_equal
import scipy.misc, scipy.io
import patchShow

caffe.set_mode_cpu()

def save_image(img, name):
    img = img[:,::-1, :, :] # Convert from BGR to RGB
    normalized_img = patchShow.patchShow_single(img, in_range=(-120,120))
    scipy.misc.imsave(name, normalized_img)

def get_shape(data_shape):
    if len(data_shape) == 4:
        return (data_shape[2], data_shape[3])
    else:
        raise Exception("Data shape invalid.")

np.random.seed(0)

generator = caffe.Net(settings.generator_definition, settings.generator_weights, caffe.TEST)
shape = generator.blobs["feat"].data.shape
generator_output_shape = generator.blobs["deconv0"].data.shape
mean = np.float32([104.0, 117.0, 123.0])
nsfw_net = caffe.Classifier("nets/open_nsfw/deploy.prototxt",
            		       "nets/open_nsfw/resnet_50_1by2_nsfw.caffemodel",
                       mean = mean,            # ImageNet mean
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

def grad_classifier(classifier, end_layer, imagein, z):
    net_dst = classifier.blobs[end_layer]
    acts = classifier.forward(data=imagein, end=end_layer)
    net_dst.diff[:] = z
    g = classifier.backward(start=end_layer, diffs=['data'])['data'][0]
    net_dst.diff.fill(0.)
    return g, acts

def grad(classifier, end_layer, i, code):
    generated = generator.forward(feat=code)
    image = crop(classifier, generated["deconv0"])
    z = np.zeros_like(classifier.blobs[end_layer].data)
    z.flat[i] = 1
    g, acts = grad_classifier(classifier, end_layer, image, z)
    generator.blobs['deconv0'].diff[...] = pad(classifier, g)
    gx = generator.backward(start='deconv0')
    generator.blobs['deconv0'].diff.fill(0.)
    return gx['feat'], image

def crop(classifier, image):
    data_shape  = classifier.blobs['data'].data.shape
    image_size  = get_shape(data_shape)
    output_size = get_shape(generator_output_shape)
    topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)
    return image.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]

def pad(classifier, image):
    data_shape  = classifier.blobs['data'].data.shape
    image_size  = get_shape(data_shape)
    output_size = get_shape(generator_output_shape)
    topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)
    o = np.zeros(generator_output_shape)
    o[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = image
    return o

def get_code(path, layer):
    batch_size = 1
    image_size = (3, 227, 227)
    images = np.zeros((batch_size,) + image_size, dtype='float32')
    in_image = scipy.misc.imread(path)
    in_image = scipy.misc.imresize(in_image, (image_size[1], image_size[2]))
    for ni in range(images.shape[0]):
        images[ni] = np.transpose(in_image, (2, 0, 1))
    data = images[:,::-1]
    matfile = scipy.io.loadmat('ilsvrc_2012_mean.mat')
    image_mean = matfile['image_mean']
    topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
    image_mean = image_mean[topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
    del matfile
    data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR
    encoder = caffe.Net(settings.encoder_definition, settings.encoder_weights, caffe.TEST)
    encoder.forward(data=data)
    feat = np.copy(encoder.blobs[layer].data)
    del encoder
    zero_feat = feat[0].copy()[np.newaxis]
    return zero_feat, data

opt_layer = 'fc6'
total_iters = 300
alpha = 1

def main(filename, iters=total_iters):
    np.random.seed(0)

    code, start_image = get_code(filename, opt_layer)
    upper_bound = np.loadtxt("act_range/3x/fc6.txt", delimiter=' ', usecols=np.arange(0, 4096), unpack=True)
    upper_bound = upper_bound.reshape(4096)

    lower_bound = np.zeros(4096)

    for i in range(0,iters):
        step_size = (alpha + (1e-10 - alpha) * i) / iters
        gn, image = grad(nsfw_net, 'prob', 1, code)

        g = 1500 * gn

        if norm(g) <= 1e-8:
            break

        code = code - step_size*g/np.abs(g).mean()
        code = np.maximum(code, lower_bound)

        # 1*upper bound produces realistic looking images
        # No upper bound produces dramatic high saturation pics
        # 1.5* Upper bound is a decent choice
        code = np.minimum(code, 1.5*upper_bound)

        save_image(image, "output/" + str(i) + ".jpg")

if __name__ == '__main__':
    main('jordan1.jpg')
