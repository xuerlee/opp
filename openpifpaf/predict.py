"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from PyQt5.QtWidgets import QApplication

from openpifpaf import decoder, logger, network, show, visualizer, __version__
from openpifpaf.predictor import Predictor



LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.predict',
        usage='%(prog)s [options] images',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)  # 初始化网络 直接在network的函数中定义了, Predictor 会自动从配置好的 network 和 decoder 中获取设置
    Predictor.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

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

    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)  # args in the predictor class
    show.configure(args)  # 画的粗细字体颜色等
    visualizer.configure(args)  # 如何显示出来

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
    annotation_painter = show.AnnotationPainter()

    predictor = Predictor(  # 推理
        visualize_image=(args.show or args.image_output is not None),
        visualize_processed_image=args.debug,
    )  # in predictor class, network.Factory().factory(head_metas=head_metas) load models from weights
    # Predictor 会自动从配置好的 network 和 decoder 中获取设置
    '''
    全局工厂模式：
    network.Factory 是一个工厂类，其配置是全局的。
    当 network.Factory.configure(args) 执行时，它修改了 Factory 类的全局属性，例如 checkpoint 或其他相关参数。
    之后，当 Predictor 实例化时，调用 network.Factory().factory() 会使用全局配置，返回已经按照 args 配置好的模型。
    network folder: init.py + factory.py(class Factory)
    '''
    for pred, _, meta in predictor.images(args.images):  # 预测结果显示
        '''  
        yield循环：
        predictor.images (ImageList, preprocess) 
        output from the called iterator 'predictor.dataset' which generated a Dataloader
        output from the called iterator 'predictor.dataloader(Dataloader)'
        output from the called iterator 'predictor.enumerated_dataloader(enumerate(Dataloader))' with a for iterator which output pred, gt, meta
        
        the input images are precessed, packed by Dataloader, and enumerated by enumerated_dataloader
        finally output a packed list with pred, gt, meta, which can be iterately called by 'next' method
        
        if the batch size is 1, only put 1 image in to the dataloader and generate a dataloader with only 1 image there.
        
        make predicting with batch size and the decoding process easier
        '''
        # json output
        if args.json_output is not None:
            json_out_name = out_name(
                args.json_output, meta['file_name'], '.predictions.json')
            LOG.debug('json output = %s', json_out_name)
            with open(json_out_name, 'w') as f:
                json.dump([ann.json_data() for ann in pred], f)

        # image output
        if args.show or args.image_output is not None:
            ext = show.Canvas.out_file_extension
            image_out_name = out_name(
                args.image_output, meta['file_name'], '.predictions.' + ext)
            LOG.debug('image output = %s', image_out_name)
            image = visualizer.Base.image()
            with show.image_canvas(image, image_out_name) as ax:
                annotation_painter.annotations(ax, pred)


if __name__ == '__main__':
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # app = QApplication(sys.argv)
    # sys.exit(app.exec_())
    main()
