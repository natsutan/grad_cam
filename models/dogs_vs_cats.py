import os
from argparse import ArgumentParser

from vgg16_transfer.vgg16_transfer import transfer_model

# 動作確認用
EPOCH = 1


def get_option():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-m', '--model', help='model: vgg16_transfer')
    arg_parser.add_argument('-e', '--epoch', type=int, default=EPOCH, help='epoch number')
    return arg_parser.parse_args()


def select_model(model_name):
    if model_name == 'vgg16_transfer':
        return transfer_model()
    else:
        print("error model name = ", model_name)
        os.sys.exit(1)


def main():
    args = get_option()
    print("model = ", args.model)
    print("epoch = ", args.epoch)
    model = select_model(args.model)


if __name__ == '__main__':
    main()
