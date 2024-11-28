"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os
import openpifpaf.transforms as transforms
import PIL
import numpy as np
import matplotlib.pyplot as plt
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
        usage='%(prog)s [options] input_folder output_folder',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    logger.cli(parser)
    network.Factory.cli(parser)  # 初始化网络 直接在network的函数中定义了


    # parser.add_argument('images', default=['pred_picture_joint_shuffle2k16/cad4.jpeg'], nargs='*',
    #                     help='input images')
    parser.add_argument('input_folder', default='/home/jiqqi/data/new-new-collective/img_with_anns', nargs='*', help='Input folder containing images')  # nargs='*' 允许该参数接受零个或多个值，使得输入的参数变得非必需。
    parser.add_argument('output_folder', default='/home/jiqqi/data/new-new-collective/img_with_anns_fm', nargs='*', help='Output folder to save feature maps' )
    # parser.add_argument('--glob',
    #                     help='glob expression for input images (for many images)')
    # parser.add_argument('-o', '--image-output', default=True, nargs='?', const=True,
    #                     help='Whether to output an image, '
    #                          'with the option to specify the output path or directory')
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
    # if args.glob:
    #     args.images += glob.glob(args.glob)
    # if not args.images:
    #     raise Exception("no image files given")

    return args



def extract_and_save_feature_maps(model, input_folder, output_folder, device):
    """Extract and save feature maps from images in the input folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    rescale_t = None
    # if long_edge:
    #     rescale_t = transforms.RescaleAbsolute(self.long_edge, fast=self.fast_rescaling)
    pad_t = None
    # if batch_size > 1:
    #     assert self.long_edge, '--long-edge must be provided for batch size > 1'
    #     pad_t = transforms.CenterPad(long_edge)
    # else:
    #     pad_t = transforms.CenterPadTight(16)
    transform = transforms.Compose_FM([
        # rescale_t,  # resize according to long edge
        transforms.CenterPadTight_FM(16),
        transforms.FM_TRANSFORM,
    ])

    for image_path in glob.glob(os.path.join(input_folder, '*')):
        try:
            with open(image_path, 'rb') as f:
                image = PIL.Image.open(f).convert('RGB')
                input_tensor = transform(image)[0].unsqueeze(0).to(device)

            # Extract features
            with torch.no_grad():
                feature_map = model.base_net(input_tensor)
            '''
            print(feature_map.size()) torch.Size([1, 1392, 31, 46])
            '''
            # feature = feature_map.cpu().numpy()[0]
            # feature = np.sum(feature, axis=0)
            # plt.imshow(feature)
            # plt.show()

            # Convert feature map to CPU and save
            feature_map = feature_map.cpu().numpy()

            feature_map_save_path = os.path.join(
                output_folder, os.path.basename(image_path).replace('.jpg', '_features.pt')
            )
            torch.save(feature_map, feature_map_save_path)
            LOG.info(f"Saved feature map to {feature_map_save_path}")
        #
        except Exception as e:
            LOG.error(f"Failed to process image {image_path}: {e}")



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
