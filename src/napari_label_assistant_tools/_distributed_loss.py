"""Differentiable global sums used only by opt-in distributed training."""
import torch


class _GlobalSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        result = value.clone()
        torch.distributed.all_reduce(result)
        return result

    @staticmethod
    def backward(ctx, gradient):
        result = gradient.contiguous().clone()
        torch.distributed.all_reduce(result)
        return result


global_sum = _GlobalSum.apply
