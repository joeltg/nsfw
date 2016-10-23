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

# caffe.set_mode_gpu()
caffe.set_mode_cpu()

def save_image(img, name):
    '''
    Normalize and save the image.
    '''
    img = img[:,::-1, :, :] # Convert from BGR to RGB
    normalized_img = patchShow.patchShow_single(img, in_range=(-120,120))
    scipy.misc.imsave(name, normalized_img)


def get_shape(data_shape):

    # Return (227, 227) from (1, 3, 227, 227) tensor
    if len(data_shape) == 4:
        return (data_shape[2], data_shape[3])
    else:
        raise Exception("Data shape invalid.")

np.random.seed(0)

generator = caffe.Net(settings.generator_definition, settings.generator_weights, caffe.TEST)
shape = generator.blobs["feat"].data.shape
# shape is a tuple = (1, 4096)
generator_output_shape = generator.blobs["deconv0"].data.shape

mean = np.float32([104.0, 117.0, 123.0])

net = caffe.Classifier("nets/placesCNN/places205CNN_deploy_updated.prototxt",
                       "nets/placesCNN/places205CNN_iter_300000.caffemodel",
                       mean = mean,            # ImageNet mean
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

nsfw_net = caffe.Classifier("../open_nsfw/nsfw_model/deploy.prototxt",
            		       "../open_nsfw/nsfw_model/resnet_50_1by2_nsfw.caffemodel",
                       mean = mean,            # ImageNet mean
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB


def grad_classifier(classifier, end_layer, imagein, z):

    net_dst = classifier.blobs[end_layer]

    # Do forward pass
    acts = classifier.forward(data=imagein, end=end_layer)

    # Do backwards pass
    net_dst.diff[:] = z
    g = classifier.backward(start=end_layer, diffs=['data'])['data'][0]

    # Cleanup
    net_dst.diff.fill(0.)
    return g, acts

def grad(classifier, end_layer, i, code):

    # Perform Forward Step
    generated = generator.forward(feat=code)
    image = crop(classifier, generated["deconv0"])

    # Set the inner product the gradient is taken w.r. to
    z = np.zeros_like(classifier.blobs[end_layer].data)
    z.flat[i] = 1

    # Do backwards step
    g, acts = grad_classifier(classifier, end_layer, image, z)
    generator.blobs['deconv0'].diff[...] = pad(classifier, g)
    gx = generator.backward(start='deconv0')

    # Cleanup
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



np.random.seed(200)

def get_code(path, layer):
  '''
  Push the given image through an encoder to get a code.
  '''

  # set up the inputs for the net:
  batch_size = 1
  image_size = (3, 227, 227)
  images = np.zeros((batch_size,) + image_size, dtype='float32')

  in_image = scipy.misc.imread(path)
  in_image = scipy.misc.imresize(in_image, (image_size[1], image_size[2]))

  for ni in range(images.shape[0]):
    images[ni] = np.transpose(in_image, (2, 0, 1))

  # Convert from RGB to BGR
  data = images[:,::-1]

  # subtract the ImageNet mean
  matfile = scipy.io.loadmat('ilsvrc_2012_mean.mat')
  image_mean = matfile['image_mean']
  topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
  image_mean = image_mean[topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
  del matfile
  data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR

  # initialize the encoder
  encoder = caffe.Net(settings.encoder_definition, settings.encoder_weights, caffe.TEST)

  # run encoder and extract the features
  encoder.forward(data=data)
  feat = np.copy(encoder.blobs[layer].data)
  del encoder

  zero_feat = feat[0].copy()[np.newaxis]

  return zero_feat, data

init_file = "images/green_pepper.jpg"
opt_layer = 'fc6'
code, start_image = get_code(init_file, opt_layer)
# code = np.random.normal(0, 1, shape)
# code is numpy array of shape (1, 4096) = [[1, 0, ... ]]

# total_iters h= 300
total_iters = 100

alpha = 1
# Load the activation range
upper_bound = lower_bound = None

# Set up clipping bounds
upper_bound = np.loadtxt("act_range/3x/fc6.txt", delimiter=' ', usecols=np.arange(0, 4096), unpack=True)
upper_bound = upper_bound.reshape(4096)
# upper_bound is a numpy array of shape (4096,)

# Lower bound of 0 due to ReLU
lower_bound = np.zeros(4096)
# lower_bound is also a numpy array of shape (4096,0)

category = 0

for i in range(0,total_iters):
    step_size = (alpha + (1e-10 - alpha) * i) / total_iters
    # step_size = 0.5
    gp, image = grad(net, 'fc8', category, code)
    gp = gp.copy()
    gn, image = grad(nsfw_net, 'prob', 1, code)

    # gp is a numpy array of shape (1, 4096)
    # image is a nupmy array of shape (1, 3, 227, 227)

    # To generate NSFW Samples
    # g =  1500*gn + 0.00000000001 * gp
    g = 1500*gn

    # To generate Regular Samples
    # g = 1*gp + 0*gn

    # print norm(gp), norm(gn)
    print norm(g)

    # if norm(g) <= 1e-8:
    #     break
    code = code - step_size * g / np.abs(g).mean()
    code = np.maximum(code, lower_bound)

    # 1*upper bound produces realistic looking images
    # No upper bound produces dramatic high saturation pics
    # 1.5* Upper bound is a decent choice
    code = np.minimum(code, 1.5*upper_bound)

    if i % 10 == 0 or True:
        print "saving image"
        save_image(image, "final_places/result-%d.jpg" % i)
