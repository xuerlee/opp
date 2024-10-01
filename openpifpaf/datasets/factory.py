from .module import DataModule
from .multiloader import MultiLoader
from .multimodule import MultiDataModule

DATAMODULES = {}


def factory(dataset):
    if '-' in dataset:
        datamodules = [factory(ds) for ds in dataset.split('-')]
        # 此处的datamodule是单个DATAMODULES[dataset]，可以有多个datamodule(如cocokp-cocodet，指coco数据集上的keppoint和detection同时做)
        # define heads according to datasets, DATAMODULES[datasets], DATAMODULES are registered in plugins/__init__
        return MultiDataModule(datamodules)  # 同下, MultiDataModule中把两个plugin拼接起来,可以加载两个数据集，定义数据集跟定义网络无关

    if dataset not in DATAMODULES:
        raise Exception('dataset {} unknown'.format(dataset))
    return DATAMODULES[dataset]()
    # 在每一个plugin中的__init__中定义了：如openpifpaf.DATAMODULES['cocodet'] = CocoDet 于是这里就可以直接调用cocodet.py，加载数据集


def cli(parser):
    group = parser.add_argument_group('generic data module parameters')
    # group.add_argument('--dataset', default='cocokp-cocodet')
    group.add_argument('--dataset', default='cocojoint')
    # group.add_argument('--dataset', default='cocodet')
    group.add_argument('--loader-workers',
                       default=None, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size',
                       default=DataModule.batch_size, type=int,  # 这里才是真正加载batch的图片多少
                       help='batch size')
    # group.add_argument('--dataset-weights', default=[1.0, 1.0], nargs='+', type=float,
    group.add_argument('--dataset-weights', default=None, nargs='+', type=float,
                       help='n-1 weights for the datasets')

    for dm in DATAMODULES.values():
        dm.cli(parser)


def configure(args):
    DataModule.set_loader_workers(args.loader_workers if not args.debug else 0)
    DataModule.batch_size = args.batch_size
    MultiLoader.weights = args.dataset_weights  # call multiloader.py

    for dm in DATAMODULES.values():
        dm.configure(args)
