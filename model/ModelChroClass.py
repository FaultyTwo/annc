from annc.res_comp.ResChroClass import ResChroClass
from annc.inception_comp.InceptionChroClass import InceptionChroClass

class ModelChroClass():
    @classmethod
    def chro_str2chunk(self, gene:str):
        """
        Turns a chromosome string into chunk layers (list)

        Args:
            gene (str): A single chromosome in string format
        
        Returns:
            chunk (list): A list of chunk layers
        """
        chunk = ResChroClass.chro_str2chunk(gene)
        new_chunk = list()
        for x in chunk:
            if 'I' not in x:
                new_chunk.append(x)
            else:
                temp = InceptionChroClass.chro_str2chunk(x)
                for y in temp:
                    new_chunk.append(y)
        return new_chunk