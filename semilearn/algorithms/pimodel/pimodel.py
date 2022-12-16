# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch 
import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool


class PiModel(AlgorithmBase):
    """
        Pi-Model algorithm (https://arxiv.org/abs/1610.02242).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.domain_align = args.domain_align if hasattr(args, 'domain_align') else False
        self.da_len = args.da_len if hasattr(args, 'da_len') else 0

        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(unsup_warm_up=args.unsup_warm_up)
    
    def init(self, unsup_warm_up=0.4):
        self.unsup_warm_up = unsup_warm_up 

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        # inference and calculate sup/unsup losses
        with self.amp_cm():

            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            # calculate BN only for the first batch
            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb_w = self.model(x_ulb_w)
            logits_x_ulb_w = outs_x_ulb_w['logits']
            outs_x_ulb_s = self.model(x_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            self.bn_controller.unfreeze_bn(self.model)

            if self.registered_hook("DomainAlignHook"):
                probs_x_lb = torch.softmax(logits_x_lb, dim=-1)
                probs_x_ulb_w = self.call_hook("dist_align", "DomainAlignHook", probs_x_ulb=probs_x_ulb_w.detach(), probs_x_lb=probs_x_lb.detach())

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          torch.softmax(logits_x_ulb_w.detach(), dim=-1),
                                          'mse')
            # TODO: move this into masking
            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        if self.registered_hook("DomainAlignHook"):
            save_dict['p_model'] = self.hooks_dict['DomainAlignHook'].p_model.cpu()
            save_dict['p_target'] = self.hooks_dict['DomainAlignHook'].p_target.cpu()
        return save_dict
    
    def load_model(self, load_path):
        checkpoint =  super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        if self.registered_hook("DomainAlignHook"):
            self.hooks_dict['DomainAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
            self.hooks_dict['DomainAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
        ]