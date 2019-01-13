from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16


def transfer_model():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


if __name__ == '__main__' :
    model = transfer_model()
    model.summary()


