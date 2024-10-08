import argparse
import logging
from openpifpaf import headmeta
from . import components
from .composite import CompositeLoss, CompositeLossByComponent
from .multi_head import MultiHeadLoss, MultiHeadLossAutoTuneKendall, MultiHeadLossAutoTuneVariance

LOG = logging.getLogger(__name__)

#: headmeta class to Loss class
# 通过 headmeta.Cif: CompositeLoss, # headmeta.Caf: CompositeLoss等可以在CompositeLoss和Conponents中通过head_meta设置损失函数。
LOSSES = {
    'a': CompositeLossByComponent,  # TODO hack: added to trigger cli() and configure()
    headmeta.Cif: CompositeLoss,
    headmeta.Caf: CompositeLoss,
    headmeta.CifDet: CompositeLoss,  # 根据headmeta的n_vector n_scale等每个head设置对应loss， 共用一个composite函数
    headmeta.TSingleImageCif: CompositeLoss,
    headmeta.TSingleImageCaf: CompositeLoss,
    headmeta.Tcaf: CompositeLoss,
}
LOSS_COMPONENTS = {
    components.Bce,
    components.SmoothL1,
    components.Scale,
    components.Laplace,
}


class Factory:
    # lambdas = [2, 1, 1]
    lambdas = [1, 1, 1] # cocodet-cocokp
    # lambdas = [1, 1] # cocokp
    # lambdas = [1] # cocodet
    component_lambdas = None
    auto_tune_mtl = False
    auto_tune_mtl_variance = False

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('losses')
        group.add_argument('--lambdas', default=cls.lambdas, type=float, nargs='+',
                           help='prefactor for head losses by head')  # 每个head的比重?
        group.add_argument('--component-lambdas',
                           default=cls.component_lambdas, type=float, nargs='+',
                           help='prefactor for head losses by component')  # loss中每个部分的比重？  在后续代码中两个合并了，以component_lambdas传入loss函数中
        assert not cls.auto_tune_mtl
        group.add_argument('--auto-tune-mtl', default=False, action='store_true',
                           help=('[experimental] use Kendall\'s prescription for '
                                 'adjusting the multitask weight'))
        assert not cls.auto_tune_mtl_variance
        group.add_argument('--auto-tune-mtl-variance', default=False, action='store_true',
                           help=('[experimental] use Variance prescription for '
                                 'adjusting the multitask weight'))
        assert MultiHeadLoss.task_sparsity_weight == \
            MultiHeadLossAutoTuneKendall.task_sparsity_weight
        assert MultiHeadLoss.task_sparsity_weight == \
            MultiHeadLossAutoTuneVariance.task_sparsity_weight
        group.add_argument('--task-sparsity-weight',
                           default=MultiHeadLoss.task_sparsity_weight, type=float,
                           help='[experimental]')

        for l in set(LOSSES.values()):
            l.cli(parser)
        for lc in LOSS_COMPONENTS:
            lc.cli(parser)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.lambdas = args.lambdas  # 各个loss的比重
        cls.component_lambdas = args.component_lambdas
        cls.auto_tune_mtl = args.auto_tune_mtl
        cls.auto_tune_mtl_variance = args.auto_tune_mtl_variance

        # MultiHeadLoss
        MultiHeadLoss.task_sparsity_weight = args.task_sparsity_weight
        MultiHeadLossAutoTuneKendall.task_sparsity_weight = args.task_sparsity_weight
        MultiHeadLossAutoTuneVariance.task_sparsity_weight = args.task_sparsity_weight

        for l in set(LOSSES.values()):
            l.configure(args)
        for lc in LOSS_COMPONENTS:
            lc.configure(args)

    def factory(self, head_metas):
        sparse_task_parameters = None
        # TODO
        # if MultiHeadLoss.task_sparsity_weight:
        #     sparse_task_parameters = []
        #     for head_net in head_nets:
        #         if getattr(head_net, 'sparse_task_parameters', None) is not None:
        #             sparse_task_parameters += head_net.sparse_task_parameters
        #         elif hasattr(head_net, 'conv'):
        #             sparse_task_parameters.append(head_net.conv.weight)
        #         else:
        #             raise Exception('unknown l1 parameters for given head: {} ({})'
        #                             ''.format(head_net.meta.name, type(head_net)))

        losses = [LOSSES[meta.__class__](meta) for meta in head_metas] # define losses
        '''
        head_metas: settings of Cif..., Caf..., Cifdet...
        '''
        '''
        [CompositeLoss(
          (bce_loss): BceL2((soft_clamp): SoftClamp())
          (scale_loss): Scale((soft_clamp): SoftClamp())
          (soft_clamp): SoftClamp()), 
          CompositeLoss(
          (bce_loss): BceL2((soft_clamp): SoftClamp())
          (scale_loss): Scale((soft_clamp): SoftClamp())
          (soft_clamp): SoftClamp()), 
          CompositeLoss(
          (bce_loss): BceL2((soft_clamp): SoftClamp())
          (scale_loss): Scale((soft_clamp): SoftClamp())
          (soft_clamp): SoftClamp())]
        '''
        # composite：设置cif和caf的损失函数； conponents：设置具体的损失函数
        component_lambdas = self.component_lambdas  # 不同的head
        if component_lambdas is None and self.lambdas is not None:
            assert len(self.lambdas) == len(head_metas)
            component_lambdas = [
                head_lambda
                for loss, head_lambda in zip(losses, self.lambdas)
                for _ in loss.field_names  # cocokp.cif, cocokp.caf, cocokp.dcaf, cocodet.cifdet  3x3=9?
            ]  # 会变成[2, 2, 2, 1, 1, 1, 1, 1, 1]
            # ['cocodet.cifdet.c', 'cocodet.cifdet.vec', 'cocodet.cifdet.scales', 'cocokp.cif.c', 'cocokp.cif.vec', 'cocokp.cif.scales', 'cocokp.caf.c', 'cocokp.caf.vec', 'cocokp.caf.scales']


        if self.auto_tune_mtl:
            loss = MultiHeadLossAutoTuneKendall(
                losses, component_lambdas, sparse_task_parameters=sparse_task_parameters)
        elif self.auto_tune_mtl_variance:
            loss = MultiHeadLossAutoTuneVariance(
                losses, component_lambdas, sparse_task_parameters=sparse_task_parameters)
        else:
            loss = MultiHeadLoss(losses, component_lambdas)  # 都会通过这里来定义
            # losses: compositeloss, loss: modulelist
        return loss
