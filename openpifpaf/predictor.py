import argparse
import logging

import PIL
import torch

from . import datasets, decoder, network, transforms, visualizer

LOG = logging.getLogger(__name__)


class Predictor:  # 推理
    """Convenience class to predict from various inputs with a common configuration."""

    batch_size = 1  #: batch size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  #: device
    fast_rescaling = True  #: fast rescaling
    loader_workers = None  #: loader workers
    long_edge = None  #: long edge

    def __init__(self, checkpoint=None, head_metas=None, *,
                 json_data=False,
                 visualize_image=False,
                 visualize_processed_image=False):
        if checkpoint is not None:
            network.Factory.checkpoint = checkpoint
        self.json_data = json_data
        self.visualize_image = visualize_image
        self.visualize_processed_image = visualize_processed_image

        self.model_cpu, _ = network.Factory().factory(head_metas=head_metas)
        self.model = self.model_cpu.to(self.device)  # 定义网络
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
            self.model.base_net = self.model_cpu.base_net
            self.model.head_nets = self.model_cpu.head_nets

        self.preprocess = self._preprocess_factory()
        self.processor = decoder.factory(self.model_cpu.head_metas)  # predictor调用factory时自动返回所需的decoder函数，processor有batch函数可以调用
        # output bboxes
        self.last_decoder_time = 0.0
        self.last_nn_time = 0.0
        self.total_nn_time = 0.0
        self.total_decoder_time = 0.0
        self.total_images = 0

        LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
                 self.device, torch.cuda.is_available(), torch.cuda.device_count())

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser, *,
            skip_batch_size=False, skip_loader_workers=False):
        """Add command line arguments.

        When using this class together with datasets (e.g. in eval),
        skip the cli arguments for batch size and loader workers as those
        will be provided via the datasets module.
        """
        group = parser.add_argument_group('Predictor')

        if not skip_batch_size:
            group.add_argument('--batch-size', default=cls.batch_size, type=int,
                               help='processing batch size')

        if not skip_loader_workers:
            group.add_argument('--loader-workers', default=cls.loader_workers, type=int,
                               help='number of workers for data loading')

        group.add_argument('--long-edge', default=cls.long_edge, type=int,
                           help='rescale the long side of the image (aspect ratio maintained)')
        group.add_argument('--precise-rescaling', dest='fast_rescaling',
                           default=True, action='store_false',
                           help='use more exact image rescaling (requires scipy)')
        group.add_argument('--volleyball-filter',
                           default=False,
                           help='filter the detected persons out of the field')

    @classmethod  # Predictor.configure(args)
    def configure(cls, args: argparse.Namespace):
        """Configure from command line parser."""
        cls.batch_size = args.batch_size
        cls.device = args.device
        cls.fast_rescaling = args.fast_rescaling
        cls.loader_workers = args.loader_workers
        cls.long_edge = args.long_edge
        cls.volleyball_filter = args.volleyball_filter  # same as 'self.'

    def _preprocess_factory(self):  # 图像处理
        rescale_t = None
        if self.long_edge:
            rescale_t = transforms.RescaleAbsolute(self.long_edge, fast=self.fast_rescaling)

        pad_t = None
        if self.batch_size > 1:
            assert self.long_edge, '--long-edge must be provided for batch size > 1'
            pad_t = transforms.CenterPad(self.long_edge)
        else:
            pad_t = transforms.CenterPadTight(16)

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            rescale_t,
            pad_t,
            transforms.EVAL_TRANSFORM,
        ])

    def dataset(self, data):
        """Predict from a dataset."""
        loader_workers = self.loader_workers
        if loader_workers is None:
            loader_workers = self.batch_size if len(data) > 1 else 0

        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.device.type != 'cpu',
            num_workers=loader_workers,
            collate_fn=datasets.collate_images_anns_meta)

        yield from self.dataloader(dataloader)  # -> enumerated_dataloader -> processor -> decoder

    def enumerated_dataloader(self, enumerated_dataloader):  # meta info is in dataloader
        """Predict from an enumerated dataloader."""
        for batch_i, item in enumerated_dataloader:
            if len(item) == 3:  # eval, processed by _eval_preprocess()
                processed_image_batch, gt_anns_batch, meta_batch = item
                image_batch = [None for _ in processed_image_batch]
            elif len(item) == 4:  # predict, processed by def images(self, file_names, **kwargs) in this class
                image_batch, processed_image_batch, gt_anns_batch, meta_batch = item
            if self.visualize_processed_image:
                visualizer.Base.processed_image(processed_image_batch[0])

            pred_batch = self.processor.batch(self.model, processed_image_batch, device=self.device)  # 调用decoder.py, output a list of cls obj in annotation.py
            self.last_decoder_time = self.processor.last_decoder_time
            self.last_nn_time = self.processor.last_nn_time
            self.total_decoder_time += self.processor.last_decoder_time
            self.total_nn_time += self.processor.last_nn_time
            self.total_images += len(processed_image_batch)

            # un-batch
            for image, pro_image, pred, gt_anns, meta in \
                    zip(image_batch, processed_image_batch, pred_batch, gt_anns_batch, meta_batch):
                LOG.info('batch %d: %s', batch_i, meta.get('file_name', 'no-file-name'))
                # pred is also a list of cls obj (each detected person is an obj)
                # load the original image if necessary
                if self.visualize_image:
                    visualizer.Base.image(image, meta=meta)

                # filter the persons out of the filed
                if self.volleyball_filter:
                    pred_ini = pred
                    pred = []
                    for ann in pred_ini:
                        if ann.__class__.__name__ == 'AnnotationDet':
                            w, h = meta['width_height'][0], meta['width_height'][1]
                            pro_h, pro_w = pro_image.size(1), pro_image.size(2)
                            h_ratio = h / pro_h
                            w_ratio = w / pro_w
                            ann_data = ann.json_data()
                            bbox = ann_data['bbox']
                            if h == 720:
                                if (h_ratio * (bbox[1] + bbox[3]) > 435) and (h_ratio * (bbox[1] + bbox[3]) < 680):
                                    pred.append(ann)
                            elif h == 1080:
                                if (h_ratio * (bbox[1] + bbox[3]) > 600) and (h_ratio * (bbox[1] + bbox[3]) < 1000):
                                    pred.append(ann)
                        else:
                            pred.append(ann)
                    # print(pred)

                pred = [ann.inverse_transform(meta) for ann in pred]
                gt_anns = [ann.inverse_transform(meta) for ann in gt_anns]
                if self.json_data:  # parse the annotation obj
                    pred = [ann.json_data() for ann in pred]

                yield pred, gt_anns, meta

    def dataloader(self, dataloader):
        """Predict from a dataloader."""
        yield from self.enumerated_dataloader(enumerate(dataloader))

    def image(self, file_name):
        """Predict from an image file name."""
        return next(iter(self.images([file_name])))

    def images(self, file_names, **kwargs):
        """Predict from image file names."""
        data = datasets.ImageList(
            file_names, preprocess=self.preprocess, with_raw_image=True)
        yield from self.dataset(data, **kwargs)  # 调用dataset

    def pil_image(self, image):
        """Predict from a Pillow image."""
        return next(iter(self.pil_images([image])))

    def pil_images(self, pil_images, **kwargs):
        """Predict from Pillow images."""
        data = datasets.PilImageList(
            pil_images, preprocess=self.preprocess, with_raw_image=True)
        yield from self.dataset(data, **kwargs)

    def numpy_image(self, image):
        """Predict from a numpy image."""
        return next(iter(self.numpy_images([image])))

    def numpy_images(self, numpy_images, **kwargs):
        """Predict from numpy images."""
        data = datasets.NumpyImageList(
            numpy_images, preprocess=self.preprocess, with_raw_image=True)
        yield from self.dataset(data, **kwargs)

    def image_file(self, file_pointer):
        """Predict from an opened image file pointer."""
        pil_image = PIL.Image.open(file_pointer).convert('RGB')
        return self.pil_image(pil_image)
