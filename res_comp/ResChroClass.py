import random, annc.misc
from pathlib import Path
from typing import Union

class ResChroClass:
    '''
    Common class for processing all residual chromosomes.
    '''
    @classmethod
    def chro_str2chunk(self, gene: str) -> list:
        '''
        Turn a chromosome string into chunk layers (list)

        - gene -> A single chromosome in string format
        '''

        # there's gotta be a better way to seperate the layers than this

        chunk = list()
        cool_str = str()
        is_skip = False

        # This.. is the worst string processing code, ever
        for x in gene:
            if x == 'R':
                is_skip = True
                chunk.append(cool_str)
                cool_str = ""  # purge

            cool_str += x

            if x != 'R' and x.isdigit() == False and is_skip == True:
                is_skip = False
                cool_str = cool_str[:-1] # compensate for the extra alpha from above .. ugh
                chunk.append(cool_str)
                cool_str = ""  # purge
                cool_str += x

        chunk.append(cool_str) # append last layer because i suck at coding
        return chunk

    @classmethod
    def chro_chunk2str(self, chunk: list):
        '''
        Turn a chunk into a chromosome string

        - chunk -> A single chromosome in string format
        '''
        gene = ""
        for x in chunk:
            gene += str(x)
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

        path = Path(__file__).parent / "chromo_format/nn_res_data.json"

        return annc.misc.chro_check(gene,path)

    @classmethod
    def chro_generate(self, population: int, size: Union[set, int], output_channel: int, 
                      max_connections: Union[set, int], max_kernel = 5, last_linear_only=True):
        '''
        Generate a set of random residual chromosomes

        Args:
            population (int): Total populations of chromosomes to generate
            size: Length of all chromosomes, either set (for range) or a single integer (constant)
            output_channel (int): Desired output channels of all generated chromosome
            max_connections: Maximum number of connections per chromosome, either set (for range) or a single integer (constant)
            max_kernel (int): Maximum possible kernel size for pooling and conv layers (default: 5)
            last_linear_only (bool): -> If true, only the last layer will be the linear layer (output layer) (default: True)

        Returns:
            out (list): A list of generated chromosomes
        *Beware that randomly generated chromosomes are unreliable, only use in GA.
        '''

        is_size_range = isinstance(size, (list, tuple, set))
        if is_size_range:  # if true
            min, max = int(size[0]), int(size[1] - 1)
        else:
            min, max = int(size - 1), int(size - 1)

        is_max_range = isinstance(size, (list, tuple, set))
        if is_max_range:  # if true
            r_min, r_max = int(max_connections[0]), int(max_connections[1])
        else:
            r_min, r_max = int(max_connections), int(max_connections)

        if min < 0:
            raise Exception("Size or min value can't be less than zero!")
        
        if output_channel <= 0:
            raise Exception("Output channels can't be less than one!")

        out = list()
        # generation use mini data
        path = Path(__file__).parent / "chromo_format/nn_res_gen_mini_data.json"

        alpha = ['P', 'C'] if last_linear_only else ['P', 'C', 'F'] # Note that "R" can't connect to the linear part
        for _ in range(population):  # how many parents?
            is_valid = False
            while is_valid == False:
                chunk = list()
                for _ in range(0, random.randint(min, max)):
                    chunk.append(random.choice(alpha)) # get a string of alphabets
                if annc.misc.chro_check(chunk,path) == -1:
                    continue
                is_valid = True
                chunk = annc.misc.random_layer(chunk,output=output_channel, max_kernel=max_kernel)

                # Add residual connections here
                
                # Check for the index for a member that starts with letter 'F' so we can popularize 'R'
                max_r = random.randint(r_min,r_max)
                where_f = [n for n, l in enumerate(chunk) if l[0] == 'F'][0]
                # residual connection must NOT be near to each other
                # Ex. "C22R1R2R1R2C22", this achieves absolutely NOTHING
                # TODO: Solve the problem above
                for r in range(max_r):
                    valid = False
                    while valid == False:
                        pair_start = random.randint(1, len(chunk) - 1)
                        if last_linear_only == False: # optional last_linear_only
                            pair_end = random.randint(pair_start, where_f - 1)
                        else:
                            pair_end = random.randint(pair_start, len(chunk) - 1)
                        if pair_start == pair_end:
                            continue # not a pair, go generate it again
                        else:
                            valid = True
                    chunk.insert(pair_start, 'R' + str(r + 1))
                    chunk.insert(pair_end, 'R' + str(r + 1))

                out.append(''.join(chunk))
        return out
