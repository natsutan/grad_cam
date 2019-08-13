import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2


image_dir = 'image'
model_dir = '../models/save/fix'
output_dir = 'output'
image_size = 150

#images = ['wiki_cat1.jpg', 'wiki_cat2.jpg', 'wiki_cat3.jpg', 'wiki_cat4.jpg', 'wiki_cat5.jpg']
# images = ['wiki_dog1.jpg', 'wiki_dog2.jpg', 'wiki_dog3.jpg', 'wiki_dog4.jpg', 'wiki_dog5.jpg']
images = ['wiki_dog5.jpg', ]
#model_names = ['vgg16_transfer', 'vgg16', 'cnn5', 'cnn5_v2', 'cnn4', 'cnn3']
model_names = ['vgg16_transfer',]
#model_names = ['cnn3', 'cnn4', 'cnn5']

model_tbl = {'vgg16_transfer': 'vgg16_transfer',
             'vgg16': 'vgg16_ft',
             'cnn5': 'cnn5',
             'cnn5_v2': 'cnn5_v2',
             'cnn4': 'cnn4',
             'cnn3': 'cnn3'}

last_layer_tbl = {'cnn5': ('conv2d_4', 256),
             'cnn5_v2': ('conv2d_4', 256),
             'cnn4': ('conv2d_3', 256),
             'cnn3': ('conv2d_2', 128)}



def load_model(name):
    fpath = os.path.join(model_dir, model_tbl[name] + '.h5')
    model = tf.keras.models.load_model(fpath)
    model.summary()
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

        if model_name in ['vgg16_transfer', 'vgg16']:
            vgg16_layer = model.get_layer('vgg16')
            last_conv_layer = vgg16_layer.get_layer('block5_conv3')
            print(model.output.shape)
            model_output = vgg16_layer.output[:, 0]

            grads = K.gradients(model_output, last_conv_layer.output)[0]
            pooled_grads = K.mean(grads, axis=(0, 1, 2))

            iterate = K.function([vgg16_layer.input], [pooled_grads, last_conv_layer.output[0]])
            pooled_grads_value, conv_layer_value = iterate([x])
            last_layer_num = 512
        else:
            last_conv_layer = model.get_layer(last_layer_tbl[model_name][0])
            print(model.output.shape)
            model_output = model.output[:, 0]
            grads = K.gradients(model_output, last_conv_layer.output)[0]
            pooled_grads = K.mean(grads, axis=(0, 1, 2))

            iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
            pooled_grads_value, conv_layer_value = iterate([x])
            last_layer_num = last_layer_tbl[model_name][1]
            print('last layer = ', last_layer_tbl[model_name][0], ' num = ', last_layer_num)

        for i in range(last_layer_num):
            conv_layer_value[:, :, 1] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_value, axis=-1)
        heetmap = np.maximum(heatmap, 0)
        print("max = ", np.max(heatmap))
        print("shape = ", heatmap.shape)
        heatmap /= np.max(heatmap)

        print(model_name + image_name)
        print(heatmap)

        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        hm_name = model_name + image_name

        cv2.imwrite(os.path.join(output_dir, 'hm_' + hm_name), heatmap)
        superimposed_img = heatmap * 0.3 + img
        cv2.imwrite(os.path.join(output_dir, 'im_' + hm_name), superimposed_img)


"""
img_path = 'save/wiki_cat1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)

x = preprocess_input(x)

model = VGG16(weights='imagenet')

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3))

last_conv_layer = model.get_layer('block5_conv3')
cat_output = model.output[:,386]
grads = K.gradients(cat_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0,1,2))
grads = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_value = iterate([x])

for i in range(512):
    conv_layer_value[:, :, 1] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_value, axis=-1)
heetmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# plt.matshow(heatmap)
# plt.show()

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('output/vgg_heatmap1.png', superimposed_img)
"""

