import torch.nn as nn
import warnings, math
from collections import OrderedDict

from annc.cnn_comp.CNNChroClass import CNN_ChroClass

class CNN_Net:
    # Create a pure CNN from a chromosome

    @classmethod
    def model_build(self, chro:str, data_shape:list, 
                    first_layer:bool = True, keep_spatial = False
                    , batch_norm:bool = True):
        """
        Builds a convolutional neural network (CNN) from a chromosome string representation.

        Args:
            chro (str): A string representing the chromosome
            data_shape (list): A list of integers representing the input shape
            first_layer (bool): A boolean indicating whether the network is in the first layer group
            keep_spatial (bool): A boolean indicating whether the spatial size should be preserved
            batch_norm (bool): A boolean indicating whether batch normalization should be applied
        
        Returns:
            network, shape (tuple): A PyTorch model and its output shape size
        
        Notes:
            This is a backend method for build a CNN model.

            In the future. This method will be rewritten for cleaner and faster network building, along
            with new features too.
        """
        
        if type(chro) is str:
            chunk = CNN_ChroClass.chro_str2chunk(chro)
            
        order = OrderedDict()
        in_channel = data_shape[1]
        out_channel = int
        # print the sentence
        
        if first_layer: 
            img_size = math.floor((data_shape[2] + data_shape[3])/2) # average height and width
        else:
            img_size = data_shape[1]
        shape_output = list(data_shape)

        ConvToLinear = False
        ConvFirst = True
        PoolFirst = True

        for index,x in enumerate(chunk):
            if x[0] == 'C':
                k_size = [int(x[1]), int(x[2])]

                if ConvFirst == True:
                    out_channel,pool_mul = img_size,img_size
                    ConvFirst, PoolFirst = False, False
                else:
                    out_channel = pool_mul

                # we want to reserve spatial size in this class
                # pad = (kernel-1)/2 (odd)
                # pad = (kernel-1) if dilation = 2 (even)
                # prefer first kernel channel
                if k_size[0] != k_size[1]: # if not equal
                    warnings.warn("A layer isn't a square! Equaling them ..")
                    k_size[1] = k_size[0]

                if k_size[0] != 1:
                    # conv of 1x1 can't have dilation! or else the model won't learn!
                    pad = k_size[0] - 1 if keep_spatial else 0
                    dil = 2 if keep_spatial else 1
                else:
                    pad, dil = 0,1

                order['conv' + str(index)] = nn.Conv2d(in_channels=in_channel,
                                                       kernel_size=k_size, out_channels=out_channel,
                                                       dilation=dil, padding=pad)
                if batch_norm:
                    order['batch_norm' + str(index)] = nn.BatchNorm2d(out_channel)
                order['relu' + str(index)] = nn.ReLU()

                if keep_spatial == False:
                    c_size_h = (shape_output[2] - k_size[0]) + 1
                    c_size_w = (shape_output[3] - k_size[1]) + 1
                    shape_output[2],shape_output[3] = c_size_h,c_size_w

                shape_output[1],in_channel = out_channel,out_channel

            elif x[0] == "P":

                k_size = (int(x[1]), int(x[2]))

                if PoolFirst == True:
                    out_channel = in_channel
                    pool_mul = img_size
                    ConvFirst, PoolFirst = False, False
                
                # using max tactic
                order['pool' + str(index)] = nn.MaxPool2d(kernel_size=k_size)

                # for pooling.. uh, floor and divide by k_size[0]
                p_size_h = math.floor((shape_output[2] / k_size[1]))
                p_size_w = math.floor((shape_output[3] / k_size[0]))
                shape_output[2],shape_output[3] = p_size_h,p_size_w
                pool_mul *= 2  # out_channel multi for convolution net
                in_channel = out_channel

            elif x[0] == "F":  # linear architecture
                if ConvToLinear == False:
                    #print("final shape:",linear_shape)
                    order['flat'] = nn.Flatten()
                    order['dropout'] = nn.Dropout(0.5)# no brainer
                    # match the size of conv output with linear input
                    in_channel = shape_output[1] * \
                        shape_output[2]*shape_output[3]
                    ConvToLinear = True

                out_channel = int(x[1:])
                order['linear' + str(index)
                      ] = nn.Linear(in_channel, out_channel)
                order['relu' + str(index)] = nn.ReLU()

                in_channel = out_channel
            else:
                raise Exception("Unknown Layer Types:", x)
        # shape output pass, match match actual output size
        net = nn.Sequential(order)
        return net,shape_output
    
class CNN_Net_Inception:
    # INCEPTION SUBPATHS
    # USE WITH CARES

    @classmethod
    def model_build(self, chro:str, data_shape:list):
        """
        A backend for building a path inside an inception network

        Args:
            chro (str): A string representing the chromosome
            data_shape (list): A list of integers representing the input shape
        
        Returns:
            A tuple containing the CNN and the output shape.
        
        Notes:
            This is a backend method for build a CNN model.

            In the future. This method will be rewritten for cleaner and faster network building, along
            with new features too.
        """
        
        # subpaths don't have linear layer
        chunk = CNN_ChroClass.chro_str2chunk(chro)
        order = OrderedDict()
        in_channel = data_shape[1]
        out_channel = int
        shape_output = list(data_shape)

        # no more calculating channels
        # inception is manually targeted

        for index,x in enumerate(chunk):
            if x[0] == 'C':
                k_size = [int(x[1]), int(x[2])]
                out_channel = int(x[3:])
                # reserve spatial
                if k_size[0] != k_size[1]: # if not equal
                    warnings.warn("A layer isn't a square! Equaling them ..")
                    k_size[1] = k_size[0]
                
                pad, dil = 0,1
                if k_size[0] != 1:
                    if k_size[0] % 2 != 0: # even
                        pad = math.floor((k_size[0] - 1)/2)
                    else: # dilation = 2
                        pad = k_size[0] - 1
                        dil = 2
                else:
                    pad, dil = 0,1

                order['conv' + str(index)] = nn.Conv2d(in_channels=in_channel,
                                                       kernel_size=k_size, out_channels=out_channel,
                                                       dilation=dil, padding=pad)
                #order['batch_norm' + str(index)] = nn.BatchNorm2d(out_channel)
                order['relu' + str(index)] = nn.ReLU()

                shape_output[1],in_channel = out_channel,out_channel

            elif x[0] == "P":
                # we have problem
                # for inception to work, we need to keep the spatial size of pooling
                # note: pooling math is the same as convolution
                # calculate them as normal. but i wonder how it will look like?
                k_size = (int(x[1]), int(x[2]))

                # a little debate here
                # should we just set dilation to 2 then only use a single equation?
                # or we should just keep doing this for faster calculation?
                # need more tests ...
                pad, dilation = 0,1
                if k_size[0] % 2 != 0: # even
                    pad = math.floor((k_size[0] - 1)/2)
                else: # dilation = 2
                    pad = k_size[0] - 1
                    dilation = 2

                order['pool' + str(index)] = nn.MaxPool2d(kernel_size=k_size, stride=1, padding=pad, dilation=dilation)

                in_channel = shape_output[1]
            else:
                raise Exception("Unknown Layer Types:", x)
        # shape output pass, match match actual output size
        net = nn.Sequential(order)
        return net,shape_output