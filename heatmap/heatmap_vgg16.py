import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2


image_dir = 'image'
model_dir = 'D:/data/vgg16_for_hm'
output_dir = 'output_vgg16'
image_size = 150

images = ['wiki_dog1.jpg', 'wiki_dog2.jpg', 'wiki_dog3.jpg', 'wiki_dog4.jpg', 'wiki_dog5.jpg']
#images = ['wiki_dog1.jpg',]

model_names = ['weights.01-0.79.hdf5',
                'weights.02-0.74.hdf5',
                'weights.03-0.84.hdf5',
                'weights.04-0.87.hdf5',
                'weights.05-0.94.hdf5',
                'weights.06-0.91.hdf5',
                'weights.07-0.95.hdf5',
                'weights.08-0.94.hdf5',
                'weights.09-0.96.hdf5',
                'weights.10-0.93.hdf5',
                'weights.11-0.96.hdf5',
                'weights.12-0.97.hdf5',
                'weights.13-0.94.hdf5',
                'weights.14-0.96.hdf5',
                'weights.15-0.96.hdf5',
                'weights.16-0.96.hdf5',
                'weights.17-0.95.hdf5',
                'weights.18-0.95.hdf5',
                'weights.19-0.94.hdf5',
                'weights.20-0.96.hdf5',
                'weights.21-0.89.hdf5',
                'weights.22-0.96.hdf5',
                'weights.23-0.97.hdf5',
                'weights.24-0.96.hdf5',
                'weights.25-0.97.hdf5',
                'weights.26-0.96.hdf5',
                'weights.27-0.96.hdf5',
                'weights.28-0.91.hdf5',
                'weights.29-0.96.hdf5',
                'weights.30-0.95.hdf5']


def load_model(name):
    fpath = os.path.join(model_dir, name)
    model = tf.keras.models.load_model(fpath)
    print('load model ', fpath)
    return model


def open_image(img_path):
    print(img_path)
    img = image.load_img(img_path, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x


for model_name in model_names:
    model = load_model(model_name)
    model.summary()
    for image_name in images:
        img_path = os.path.join(image_dir, image_name)
        x = open_image(img_path)
        preds = model.predict(x)
        print('Predicted:', preds[0][0])

        vgg16_layer = model.get_layer('vgg16')
        last_conv_layer = vgg16_layer.get_layer('block5_conv3')
        print(model.output.shape)
        model_output = vgg16_layer.output[:, 0]

        grads = K.gradients(model_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        iterate = K.function([vgg16_layer.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_value = iterate([x])
        last_layer_num = 512

        for i in range(last_layer_num):
            conv_layer_value[:, :, 1] *= pooled_grads_value[i]



        heatmap = np.mean(conv_layer_value, axis=-1)
        heetmap = np.maximum(heatmap, 0)
        print("max = ", np.max(heatmap))
        heatmap /= np.max(heatmap)

        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        hm_name =  model_name + image_name

        cv2.imwrite(os.path.join(output_dir, 'hm_' + hm_name), heatmap)
        superimposed_img = heatmap * 0.3 + img
        cv2.imwrite(os.path.join(output_dir, 'im_' + hm_name), superimposed_img)


