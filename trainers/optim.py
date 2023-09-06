""" modified from dassl.optim """
import warnings
import torch
import torch.nn as nn

from dassl.optim.radam import RAdam

AVAI_OPTIMS = ['adam', 'amsgrad', 'sgd', 'rmsprop', 'radam', 'adamw']


def build_optimizer(model, optim_cfg, param_groups=None):
    optim = optim_cfg.NAME
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPNING
    sgd_nesterov = optim_cfg.SGD_NESTEROV
    rmsprop_alpha = optim_cfg.RMSPROP_ALPHA
    adam_beta1 = optim_cfg.ADAM_BETA1
    adam_beta2 = optim_cfg.ADAM_BETA2
    staged_lr = optim_cfg.STAGED_LR
    new_layers = optim_cfg.NEW_LAYERS
    base_lr_mult = optim_cfg.BASE_LR_MULT

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            f'optim must be one of {AVAI_OPTIMS}, but got {optim}'
        )

    if param_groups is not None and staged_lr:
        warnings.warn(
            'staged_lr will be ignored, if you need to use staged_lr, '
            'please bind it with param_groups yourself.'
        )

    if param_groups is None:
        if staged_lr:
            # modify the function of lr_mult
            exp = optim_cfg.LR_EXP
            lr *= exp
            base_lr_mult /= exp
            
            if not isinstance(model, nn.Module):
                raise TypeError(
                    'When staged_lr is True, model given to '
                    'build_optimizer() must be an instance of nn.Module'
                )

            if isinstance(model, nn.DataParallel):
                model = model.module

            if isinstance(new_layers, str):
                if new_layers is None:
                    warnings.warn('new_layers is empty (staged_lr is useless)')
                new_layers = [new_layers]

            base_params, new_params = [], []
            base_layers, new_layers_ = [], []

            for name, module in model.named_children():
                is_new = False

                for layer in new_layers:
                    if layer in name:
                        is_new = True
                        break
                    
                if is_new:
                    new_params += [p for p in module.parameters()]
                    new_layers_.append(name)
                else:
                    base_params += [p for p in module.parameters()]
                    base_layers.append(name)
            
            param_groups = [{'params': base_params,
                             'lr': lr * base_lr_mult}, 
                            {'params': new_params}]

            # return lr of each layer
            infos = [{'layers': base_layers, 
                      'lr': lr * base_lr_mult},
                     {'layers': new_layers_,
                      'lr': lr}]
        else:
            if isinstance(model, nn.Module):
                param_groups = model.parameters()
            else:
                param_groups = model
            
            infos = None

    if optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == 'amsgrad':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == 'radam':
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    else:
        raise NotImplementedError(f'Optimizer {optim} not implemented yet!')

    return optimizer, infos