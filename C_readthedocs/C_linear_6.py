import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,input_features,output_features,bias=True):
        self.input_features=input_features
        self.output_features=output_features
# nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.
        #神经网络。Parameter是一种特殊的变量，它会得到

#自动注册为模块的参数，一旦它被分配

#作为属性。需要注册参数和缓冲区

#它们不会出现在.parameters()中(不应用于缓冲区)，以及

#不会在调用例。cuda()时被转换。您可以使用

# .register_buffer()注册缓冲区。

#神经网络。形参永远不能是易失性的，而且与变量不同，

#它们默认要求渐变。
        self.weight =nn.Parameter(torch.Tensor(input_features,output_features))
        if bias:
            self.bias=nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            #你应该总是注册所有可能的参数，但是
            #如果你愿意，可选的选项可以是None。
            self.register_parameter('bias',None)
        # Not a very smart way to initialize weights
        #不是一个非常聪明的初始化权重的方法
        self.weight.data.uniform(-0.1,0.1)
        if bias is not None:
            self.bias.data.uniform(-0.1,0.1)

    def forward(self,input):
        #查看autograd部分，了解这里发生了什么。(# See the autograd section for explanation of what happens here.)
        return Linear()(input,self.weight,self.bias)
        #注意这个Linear是之前实现过的Linear