from annc.dict_obj.DictProcess import DictProcess
from annc.model.ModelBuilder import ModelBuilder

class DictModel:
    @classmethod
    def dict_to_model(self, gene_dict:dict, data_shape:list):
        """
        Create a model from a dict network

        Args:
            gene_dict (dict): A dictionary of neural network
            data_shape (list): The shape of the data
        
        Returns:
            model: A PyTorch model
        """
        if type(gene_dict) is not dict:
            raise Exception("Argument must be a dictionary")

        chro = DictProcess.dict_to_chromosome(gene_dict, return_type="chunk")
        # go write your ultimate model builders, resnet needs downsampling [x]
        model = ModelBuilder(chro, data_shape)
        return model