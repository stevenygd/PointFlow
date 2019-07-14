import torch
from torch.autograd import Function
# from extensions.StructuralLosses.StructuralLossesBackend import NNDistance, NNDistanceGrad
from metrics.StructuralLosses.StructuralLossesBackend import NNDistance, NNDistanceGrad

# Inherit from Function
class NNDistanceFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, seta, setb):
        #print("Match Cost Forward")
        ctx.save_for_backward(seta, setb)
        '''
        input:
	        set1 : batch_size * #dataset_points * 3
	        set2 : batch_size * #query_points * 3
        returns:
	        dist1, idx1, dist2, idx2
        '''
        dist1, idx1, dist2, idx2 = NNDistance(seta, setb)
        ctx.idx1 = idx1
        ctx.idx2 = idx2
        return dist1, dist2

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        #print("Match Cost Backward")
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        seta, setb = ctx.saved_tensors
        idx1 = ctx.idx1
        idx2 = ctx.idx2
        grada, gradb = NNDistanceGrad(seta, setb, idx1, idx2, grad_dist1, grad_dist2)
        return grada, gradb

nn_distance = NNDistanceFunction.apply

