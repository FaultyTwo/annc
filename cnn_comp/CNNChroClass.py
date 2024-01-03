from pathlib import Path
import annc.misc

class CNN_ChroClass:
    '''
    Common class for processing all layer-based chromosomes.
    '''
    @classmethod
    def chro_str2chunk(self, gene: str) -> list:
        """
        Turn a chromosome string into chunk layers (list)

        Args:
            gene (str): A single chromosome in string format
        
        Returns:
            chunk (list): A list of chunk layers
        """

        # there's gotta be a better way to seperate the layers than this

        chunk = list()
        cool_str = str()
        #print(gene)
        for x in gene:
            if x.isalpha() == True:
                chunk.append(cool_str)
                cool_str = ""  # purge
            cool_str += x
        chunk.append(cool_str) # append last layer because i suck at coding
        del chunk[0] # seriously, where did this empty element come from?
        # print(chunk)
        return chunk

    @classmethod
    def chro_chunk2str(self, chunk: list):
        """
        Turn a chunk layers (list) into a chromosome string

        Args:
            chunk (list): A list of chunk layers
        
        Returns:
            gene (str): A single chromosome in string format
        """
        gene = ""
        for x in chunk:
            gene += str(x)
        return gene

    @classmethod
    def chro_check(self, gene: str):
        """
        Check the validity of a chromosome

        Args:
            gene (str): A single chromosome in string format
        
        Returns:
            If valid:
                gene (str): A single chromosome in string format
            If invalid:
                returns -1 instead
        """

        # just in case if people keeps forgetting
        if type(gene) is list:
            gene = self.chro_chunk2str(gene)

        path = Path(__file__).parent / "chromo_format/cnn_data.json"

        return annc.misc.chro_check(gene,path)
