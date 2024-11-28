"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os
from PIL import Image
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from PyQt5.QtWidgets import QApplication

from openpifpaf import decoder, logger, network, show, visualizer, __version__
from openpifpaf.predictor import Predictor



LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.featuremap',
        usage='%(prog)s [options] images',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    logger.cli(parser)
    network.Factory.cli(parser)  # 初始化网络 直接在network的函数中定义了


    parser.add_argument('images', default=['pred_picture_joint_shuffle2k16/cad4.jpeg'], nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--image-output', default=True, nargs='?', const=True,
                        help='Whether to output an image, '
                             'with the option to specify the output path or directory')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='Whether to output a json file, '
                             'with the option to specify the output path or directory')
    parser.add_argument('--disable-cuda', default=False, action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    logger.configure(args, LOG)  # logger first

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    network.Factory.configure(args)

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    return args


def out_name(arg, in_name, default_extension):
    """Determine an output name from args, input name and extension.

    arg can be:
    - none: return none (e.g. show image but don't store it)
    - True: activate this output and determine a default name
    - string:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    if arg is None:
        return None

    if arg is True:
        return in_name + default_extension

    if os.path.isdir(arg):
        return os.path.join(
            arg,
            os.path.basename(in_name)
        ) + default_extension

    return arg
from PyQt5.QtWidgets import QApplication


def main():
    args = cli()
    model, _ = network.Factory().factory()
    model.to(args.device)
    model.eval()
    extract_and_save_feature_maps(model, args.input_folder, args.output_folder, args.device)

if __name__ == '__main__':
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # app = QApplication(sys.argv)
    # sys.exit(app.exec_())
    main()
