import torch
from torch.autograd import Function
from metrics.StructuralLosses.StructuralLossesBackend import ApproxMatch, MatchCost, MatchCostGrad

# Inherit from Function
class MatchCostFunction(Function):
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
	        match : batch_size * #query_points * #dataset_points
        '''
        match, temp = ApproxMatch(seta, setb)
        ctx.match = match
        cost = MatchCost(seta, setb, match)
        return cost

    """
    grad_1,grad_2=approxmatch_module.match_cost_grad(xyz1,xyz2,match)
	return [grad_1*tf.expand_dims(tf.expand_dims(grad_cost,1),2),grad_2*tf.expand_dims(tf.expand_dims(grad_cost,1),2),None]
	"""
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        #print("Match Cost Backward")
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        seta, setb = ctx.saved_tensors
        #grad_input = grad_weight = grad_bias = None
        grada, gradb = MatchCostGrad(seta, setb, ctx.match)
        grad_output_expand = grad_output.unsqueeze(1).unsqueeze(2)
        return grada*grad_output_expand, gradb*grad_output_expand

match_cost = MatchCostFunction.apply

