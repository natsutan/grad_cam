from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16


def transfer_model_ft(batch=None):
    model = models.Sequential()
    if batch is None:
        conv_base = VGG16(include_top=False, input_shape=(150, 150, 3))
    else:
        input_tensor = layers.Input(batch_shape=(64, 150, 150, 3))
        conv_base = VGG16(include_top=False, input_tensor=input_tensor)

    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


if __name__ == '__main__' :
    model = transfer_model()
    model.summary()


