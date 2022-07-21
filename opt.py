# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import numpy as np
import math


class LAGD(Optimizer):
    #层级优化，需要考虑过大梯度和过大参数对平均值的影响
    #betas：(动量系数，归一化系数)，eps：防除0，weight_decay：L2权重衰减，max_change：梯度截断学习率倍数0不截断，relative：是否成比例训练，r_betas：比例系数
    #layer：是否按层归一化，avg_clamp：抑制层中过大值倍数0不抑制，min_lr：相关度学习中的最小非相关学习率，防止数值过小导致学习率过小
    def __init__(self, params, lr=1e-2,min_lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, max_change=5, layer=True, avg_clamp=0, relative=True, r_betas=0.99):#在一次训练中，一个参数的最大改变量为学习率的倍数
     
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr,min_lr=min_lr, betas=betas, eps=eps, weight_decay=weight_decay, max_change=max_change, layer=layer, avg_clamp=avg_clamp, relative=relative, r_betas=r_betas)
        super(LAGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LAGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step. Arguments: closure (callable, optional): A closure that reevaluates the model and returns the loss. """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            b_m, b_n= group['betas']
            b_r=group['r_betas']
            relative=group['relative']
            eps=group['eps']
            lr=group['lr']
            min_lr=min(group['min_lr'],group['lr']*1e-3)
            max_change=group['max_change']
            avg_clamp=group['avg_clamp']
            layer=group['layer']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('LAGD does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    #标量的移动平均用平方，向量的移动平均用均值
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    if not layer:#逐个放缩移动均值使用平方，是为了防止其被小量快速中和，起放大作用
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        state['exp_avg_whole'] = 0#平方，起放大作用，会使平均梯度偏低，因此这里使用均值
                    if relative and layer:
                        state['val_avg_whole'] = 0
                    elif relative and not layer:
                        state['val_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']

                if not layer:
                    exp_avg_sq = state['exp_avg_sq']
                    if relative:
                        val_avg_sq = state['val_avg_sq']

                state['step'] += 1
               
                bias_correction_m = 1 - b_m ** state['step']
                bias_correction_n = 1 - b_n ** state['step']
                if relative:
                    bias_correction_r = 1 - b_r ** state['step']


                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                #参数过大值会导致整体改变量上涨
                if relative and layer:
                    if avg_clamp > 0:
                        temp=p.abs()
                        mean=temp.mean()
                        state['val_avg_whole'] = state['val_avg_whole'] * b_r + temp.clamp_(-mean * avg_clamp,mean * avg_clamp).mean().item() * (1 - b_r)
                    else:
                        state['val_avg_whole'] = state['val_avg_whole'] * b_r + p.abs().mean().item() * (1 - b_r)
                elif relative and not layer:
                    val_avg_sq.mul_(b_r).addcmul_(p, p, value=1 - b_r)


                exp_avg.mul_(b_m).add_(grad, alpha=1 - b_m)
                #梯度过大值会导致整体改变量下降
                if layer:
                    if avg_clamp > 0:
                        temp=grad.abs()
                        mean=temp.mean()
                        state['exp_avg_whole']=state['exp_avg_whole'] * b_n + temp.clamp_(-mean * avg_clamp,mean * avg_clamp).mean().item() * (1 - b_n)
                    else:
                        state['exp_avg_whole']=state['exp_avg_whole'] * b_n + grad.abs().mean().item() * (1 - b_n)
                    denom = (state['exp_avg_whole'] / bias_correction_n) + eps
                else:
                    exp_avg_sq.mul_(b_n).addcmul_(grad, grad, value=1 - b_n)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction_n)).add_(eps)

                if layer and max_change > 0:
                    exp_avg.clamp_(-max_change * denom * bias_correction_m, max_change * denom * bias_correction_m)

                step_size = lr / bias_correction_m

                if relative and layer:
                    #计算得出非相对值的学习率
                    step_size = step_size * max(state['val_avg_whole'] / bias_correction_r,min_lr/step_size)
                    denom=torch.ones_like(exp_avg).mul_(denom)
                elif not relative and layer:
                    denom=torch.ones_like(exp_avg).mul_(denom)
                elif relative and not layer:
                    #因为非相对值的学习率是向量，因此只能用除法加在denom上
                    denom.div_((val_avg_sq.sqrt()/math.sqrt(bias_correction_r)).add_(eps).clamp_(min=min_lr/step_size))


                p.addcdiv_(exp_avg, denom, value=-step_size)

     



        return loss
