'''Deep Dreaming in Keras.
Run the script with:
```
python deep_dream.py path_to_your_base_image.jpg prefix_for_results
```
e.g.:
```
python deep_dream.py img/mypic.jpg results/dream
```
'''
from __future__ import print_function

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import scipy
import argparse

from keras.applications import inception_v3
from keras import backend as K

parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')#create an object which includes the interpretation of commands
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')# add argument base_image_path in ArgumentParser
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')# add argument result_prefix

args = parser.parse_args()# was used to parse the parameters
base_image_path = args.base_image_path
result_prefix = args.result_prefix

# These are the names of the layers
# for which we try to maximize activation,
# as well as their weight in the final loss
# we try to maximize.
# You can tweak these setting to obtain new visual effects.
settings = {
    'features': {
        'mixed2': 0.2,#layer name and its weights
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    },
}

#Sets the learning phase to a fixed value.
K.set_learning_phase(0)

# Build the InceptionV3 network with our placeholder.
# The model will be loaded with pre-trained ImageNet weights.
model = inception_v3.InceptionV3(weights='imagenet',
                                 include_top=False)#return A Keras model instance
#返回keras模型对象
dream = model.input
print('Model loaded.')

# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])#use dict format (key:value) to fetch the elements from the array model.layers(use the format of for to get element) layers is a keras' class
#create a new dict by using the existed dict model.layers   get all layers used in the inception_v3

# Define the loss.
loss = K.variable(0.)#Instantiates a variable and returns it.
for layer_name in settings['features']:
    # Add the L2 norm of the features of a layer to the loss.
    assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.' #expression is layer_name in layer_dict.keys()      if layer_name is the 'key' to the dict layer-dict
    coeff = settings['features'][layer_name]#fectch and define respective values in setting array
    x = layer_dict[layer_name].output#get the output of the layer.
    # We avoid border artifacts by only involving non-border pixels in the loss.
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if K.image_data_format() == 'channels_first':#k.image_data_format(),
        loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
    else:
        loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# Compute the gradients of the dream wrt the loss.
grads = K.gradients(loss, dream)[0]
# Normalize gradients.
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# Set up function to retrieve the value
# of the loss and gradients given an input image.
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path) #Loads an image into PIL format. PIL Python Imaging Library  is a free library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
    #Some of the file formats supported include PPM, PNG, JPEG, GIF, TIFF, and BMP.
    img = img_to_array(img)#Converts a PIL Image instance to a Numpy array.,inputing parameter 'img' is PIL image instance, the result would be a 3D numpy array
    img = np.expand_dims(img, axis=0)#expand the dimension
    img = inception_v3.preprocess_input(img)
    return img
'''def preprocess_input(x):
  x /= 255.
  x -= 0.5
  x *= 2.
  return x'''

def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

"""Process:
- Load the original image.
- Define a number of processing scales (i.e. image shapes),
    from smallest to largest.
- Resize the original image to the smallest scale.
- For every scale, starting with the smallest (i.e. current one):
    - Run gradient ascent
    - Upscale image to the next scale
    - Reinject the detail that was lost at upscaling time
- Stop when we are back to the original size.
To obtain the detail lost during upscaling, we simply
take the original image, shrink it down, upscale it,
and compare the result to the (resized) original image.
"""






#main operation
# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.01  # Gradient ascent step size,0.01
num_octave = 3  # Number of scales at which to run gradient ascent 3
octave_scale = 1.4  # Size ratio between scales 1.4
iterations = 20  # Number of ascent steps per scale 20      adjust to 100
max_loss = 5.#10

img = preprocess_image(base_image_path)#get an inception_v3 image model
if K.image_data_format() == 'channels_first':
    original_shape = img.shape[2:]
else:
    original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)#get different shapes size during each scale
successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)#get a copy of original image
shrunk_original_img = resize_img(img, successive_shapes[0])#resize the original image

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)#resize image using each shape we got during every scale
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)#return the result of img + grad value at each iteration
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)#this shape is the final successive shape size
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img#used the shrinked and upscaled images and original image to get the loss during the upscaling

    img += lost_detail#result
    shrunk_original_img = resize_img(original_img, shape)

save_img(img, fname=result_prefix + '.png')