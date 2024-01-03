import torch.nn as nn, torch
from collections import OrderedDict

from annc.cnn_comp.CNNNet import CNN_Net
from annc.inception_comp.InceptionBlock import InceptionBlock
from annc.model.ModelChroClass import ModelChroClass

class ModelBuilder(nn.Module):
    """
    Builds a model from a chromosome.

    Args:
        network (list): A network chromosome in chunk format
        data_shape (list): The shape of the data
    
    Returns:
        model: A PyTorch model
    """

    def __init__(self, network:list, data_shape:list):
        super().__init__()
        if type(network) is str:
            network = ModelChroClass.chro_str2chunk(network)

        incept = InceptionBlock()
        shape_output = list(data_shape)
        self.networks = OrderedDict()
        self.shape_sequence = list()
        self.layers = network

        for index,x in enumerate(network):
            if x[0] != 'R': # not a connection
                if type(x) is str: # is simple cnn
                    # mutating this layer shouldn't include 'F' for obvious reason
                    boo = True if index == 0 else False
                    cnn_model,shape_output = CNN_Net.model_build(x,shape_output,boo,True)
                    self.networks['sect' + str(index)] = cnn_model
                elif type(x) is list: # is inception
                    # splitted into chunk nicely
                    # now.. to create inception block
                    incept_model,shape_output = incept.model_build(x,shape_output)
                    self.networks['sect' + str(index)] = incept_model
                else:
                    raise Exception("Coca cola exploda *BOOM*")
                self.shape_sequence.append(shape_output)
            else: # is a connection
                self.shape_sequence.append(str(x))

        self.down_networks = OrderedDict() # downsampling layers

        for idx,y in enumerate(self.shape_sequence):
            if y[0] == 'R':
                # if residual, search for the end connection, then get the shape size before it
                # search self.shape_sequence then print their indices that starts with 'R'
                try:
                    end_pair = self.shape_sequence.index(y,idx+1) # don't search itself
                except:
                    # if doesn't found the end pair, then it's the end pair
                    continue

                # get previous shape size (in case of residual connect next to each other)
                # this is because our residual generator maximum connections refer to new pairs
                cool_idx = idx # for finding data_shape object
                start_shape = self.shape_sequence[cool_idx]
                end_shape = self.shape_sequence[end_pair]

                while type(start_shape) is not list:
                    cool_idx -= 1
                    start_shape = self.shape_sequence[cool_idx]

                while type(end_shape) is not list:
                    end_pair -= 1
                    end_shape = self.shape_sequence[end_pair]

                # this method doesn't support in case of "same start point, different end points" connections
                # when there's a change in shape of data (pooling, different channels .etc)
                # we might need extra keying to account for this
                if start_shape != end_shape:
                    # TODO: Extra keying for "same start point, different end points" case
                    self.down_networks[str(y)] = DownsamplingRes(start_shape, end_shape)
            
        print(self.shape_sequence)
        self.fake_sequence = nn.ModuleDict(self.networks)
    
    def forward(self,x):
        result_dict = dict()
        for index,lay in enumerate(self.layers): # loop thru the chromosome
            if lay[0] == 'R': # if a connection is found
                if lay in result_dict: # check for end connection
                    if lay in self.down_networks:
                        res_res = self.down_networks[lay](result_dict[lay])
                        x = res_res + x
                    else:
                        x = result_dict[lay] + x # add the connection
                else:
                    result_dict[str(lay)] = x # add the result to the list
            else:
                x = self.fake_sequence["sect" + str(index)](x)
        result_dict = dict() # flush the dict
        return x

class DownsamplingRes(nn.Module):
    '''
    ("PRIVATE" METHOD)

    Conv Layer for downsampling.
    Including Conv channel + Adaptive Pooling for adapting the size to match the output
    '''
    def __init__(self, input_shape:list, output_shape:list):
        super().__init__()
        self.channel_adjust = nn.Conv2d(input_shape[1], output_shape[1], kernel_size=1, dilation=2)
        self.adapt_pool = nn.AdaptiveMaxPool2d((output_shape[2], output_shape[3]))

    def forward(self, x):
        x = self.channel_adjust(x)
        x = self.adapt_pool(x)
        return x