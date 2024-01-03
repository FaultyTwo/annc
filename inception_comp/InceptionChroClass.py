import random, annc.misc
from pathlib import Path

# TODO: Complete inception generator function
# TODO: Complete inception chromosome checking

class InceptionChroClass:
    '''
    Processing chromosome information for Inception-like classes
    Strictly for inception chromosome format only
    '''
    @classmethod
    def chro_str2chunk(self, gene: str) -> list:
        '''
        Turn a chromosome string into chunk layers (list)

        - gene -> A single chromosome in string format
        '''
        sep = gene.split('I')
        sep = ' '.join(sep).split() # die empty members, die
        #print(sep)
        for index,x in enumerate(sep):
            if x[0] == 'S': # inception part
                inception_list = x.split('S')
                del inception_list[0] # remove empty member, ''
                #print(inception_list)
                sep[index] = inception_list
        return sep

    @classmethod
    def chro_chunk2str(self, chunk: list):
        '''
        Turn a chunk into a chromosome string

        - chunk -> A single chromosome in string format
        '''
        gene = ''
        for x in chunk:
            if type(x) is str:
                gene += x
            elif type(x) is list:
                gene += 'I'
                for y in x:
                    gene += 'S' + y
                gene += 'I'
            else:
                raise Exception('Invalid chunk type for Inception network')
        return gene

    @classmethod
    def chro_check(self, gene: str):
        '''
        Check the validity of a chromosome

        - gene -> A single chromosome in string format

        If it's valid, return the input
        If it's not valid, return -1 instead
        '''

        # just in case if people keeps forgetting
        if type(gene) is list:
            gene = self.chro_chunk2str(gene)

        path = Path(__file__).parent / "have fun writing dfa for this chromosome lmao"

        return annc.misc.chro_check(gene,path)

    @classmethod
    def chro_generate(self, population: int, size, output_channel: int, type:str):
        '''
        Generate a set of random chromosomes

        ### Parameters
        - population -> total population of the chromosomes
        - size -> length of all chromosomes, either set (for range) or a single integer
        - output_channel -> desired output channels of all chromosomes
        - type -> types of inception to generate
            - "all-same": all contents inside inception blocks are the same
            - "all_diff": all contents inside inception blocks are different from each other
            - "n-gram-x": *ISN'T IMPLEMENTED YET*

        *Beware that randomly generated chromosomes are unreliable, only use in GA.

        We might need to write a random layer function for inception by our own
        This means that in the future, we might need to seperate 'misc' function for each types of networks
        '''
        pass