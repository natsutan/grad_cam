import os
import pickle
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vgg16_transfer.vgg16_transfer import transfer_model
from vgg16_ft.vgg16_fulltrain import transfer_model_ft
from cnn3.cnn3 import cnn3_model
from cnn4.cnn4 import cnn4_model
from cnn5.cnn5 import cnn5_model
import tensorflow.keras.callbacks
import tensorflow.keras.backend

# 動作確認用
EPOCH = 1
USE_TPU = False

train_dir = 'D:/data/dog_and_cat_small/train'
validation_dir = 'D:/data/dog_and_cat_small/validation'

#train_dir = '/home/natsutan0/myproj/grad_cam/data/dog_and_cat/train'
#validation_dir = '/home/natsutan0/myproj/grad_cam/data/dog_and_cat/validation'


save_path = 'save'
save_file = ''
tensorboard_path = 'log'
log_dir = ''
pickle_file = ''


def get_option():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-m', '--model', help='model: vgg16_transfer')
    arg_parser.add_argument('-e', '--epoch', type=int, default=EPOCH, help='epoch number')
    arg_parser.add_argument('-t', '--tpu', default=EPOCH, help='use tpu')

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
    elif model_name == 'vgg16':
        log_dir = os.path.join(tensorboard_path, 'vgg16_ft_log')
        pickle_file = os.path.join(save_path, 'vgg16_ft.pickle')
        save_file = os.path.join(save_path, 'vgg16_ft.h5')

        return transfer_model_ft(batch=64)
    elif model_name == 'cnn3':
        log_dir = os.path.join(tensorboard_path, 'cnn3')
        pickle_file = os.path.join(save_path, 'cnn3.pickle')
        save_file = os.path.join(save_path, 'cnn3.h5')

        return cnn3_model()
    elif model_name == 'cnn4':
        log_dir = os.path.join(tensorboard_path, 'cnn4')
        pickle_file = os.path.join(save_path, 'cnn4.pickle')
        save_file = os.path.join(save_path, 'cnn4.h5')

        return cnn4_model()
    elif model_name == 'cnn5':
        log_dir = os.path.join(tensorboard_path, 'cnn5')
        pickle_file = os.path.join(save_path, 'cnn5.pickle')
        save_file = os.path.join(save_path, 'cnn5.h5')

        return cnn5_model()
    else:
        print("error model name = ", model_name)
        os.sys.exit(1)


def main():
    global EPOCH, USE_TPU
    args = get_option()

    print("model = ", args.model)
    EPOCH = args.epoch
    print("epoch = ", EPOCH)

    if args.tpu == 'true':
        USE_TPU = True
        print("USE TPU")

    model = select_model(args.model)
    model.summary()

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
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
        history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=EPOCH,
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
