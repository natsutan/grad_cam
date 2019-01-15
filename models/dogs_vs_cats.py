import os
import pickle
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vgg16_transfer.vgg16_transfer import transfer_model
import tensorflow.keras.callbacks
import tensorflow.keras.backend

# 動作確認用
EPOCH = 1
USE_TPU = False

train_dir = 'D:/data/dog_and_cat_small/train'
validation_dir = 'D:/data/dog_and_cat_small/validation'

save_path = 'save'
save_file = ''
tensorboard_path = 'log'
log_dir = ''
pickle_file = ''


def get_option():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-m', '--model', help='model: vgg16_transfer')
    arg_parser.add_argument('-e', '--epoch', type=int, default=EPOCH, help='epoch number')
    return arg_parser.parse_args()


def select_model(model_name):
    global save_file, pickle_file, log_dir
    if model_name == 'vgg16_transfer':
        if USE_TPU:
            log_dir = os.path.join(tensorboard_path, 'vgg16_tr_tpu_log')
            pickle_file = os.path.join(save_path, 'vgg16_transfer_tpu.pickle')
            save_file = os.path.join(save_path, 'vgg16_transfer_tpu.h5')
        else:
            log_dir = os.path.join(tensorboard_path, 'vgg16_tr_log')
            pickle_file = os.path.join(save_path, 'vgg16_transfer.pickle')
            save_file = os.path.join(save_path, 'vgg16_transfer.h5')

        return transfer_model(batch=64)
    else:
        print("error model name = ", model_name)
        os.sys.exit(1)


def main():
    global EPOCH
    args = get_option()
    print("model = ", args.model)
    EPOCH = args.epoch
    print("epoch = ", EPOCH)
    model = select_model(args.model)
    model.summary()

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary'
    )

    model.compile(loss='binary_crossentropy', optimizer=tf.train.RMSPropOptimizer(1e-4), metrics=['acc'])

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    cbks = [tb_cb]

    if USE_TPU:
        tpu_grpc_url = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[os.environ['TPU_NAME']])
        strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu_grpc_url)
        tpu_model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=strategy
        )
        history = tpu_model.fit_generator(train_generator, steps_per_epoch=100, epochs=50,
                                          validation_data=validation_generator, validation_steps=50,
                                          callbacks=cbks,
                                          )

        tpu_model.save(save_file)
    else:
        history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                                      validation_data=validation_generator, validation_steps=50,
                                      callbacks=cbks
                                      )
        model.save(save_file)

    with open(pickle_file, mode='wb') as fp:
        print('save history to ', pickle_file)
        pickle.dump(history.history, fp)

    print('finish')


if __name__ == '__main__':
    main()
