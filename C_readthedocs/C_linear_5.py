
import torch

from torch.autograd.gradcheck import gradcheck
from torch.autograd.variable import Variable


def linear(input,weight,bias=None):
    return Linear()(input,weight,bias)

input=(Variable(torch.randn(20,20).double(),requires_grad=True),)
test=gradcheck(Linear(),input,eps=1e-6,atol=1e-4)
print(test)