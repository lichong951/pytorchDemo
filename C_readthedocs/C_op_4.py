import torch
from torch.autograd import Function

class Linear(Function):

    def forward(self,input,weight,bias=None):
        self.save_for_backward(input,weight,bias)
        output=input.mm(weight.t())
        if bias is not None:
            output +=bias.unsqueeze(0).expand_as(output)
            
        return output
    
    def backward(self,grad_output):
        input,weight,bias=self.saved_tensors
        grad_input=grad_weight=grad_bias=None

        if self.needs_input_grad[0]:
            grad_input=grad_output.mm(input)
        if self.needs_input_grad[1]:
            grad_weight=grad_output.t().mm(input)
        if bias is not None and self.needs_input_grad[2]:
            grad_bias=grad_output.sum(0).squeeze(0)

        return grad_input,grad_weight,grad_weight