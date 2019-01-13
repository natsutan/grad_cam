from tensorflow.keras.applications.vgg16 import VGG16
save_file = 'save/vgg16_pretrain.h5'

model=VGG16(weights='imagenet')

model.save(save_file)
