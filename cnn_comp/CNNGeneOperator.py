'''
A class for handing gene operation within VGG-like chromosomes
'''

from annc.cnn_comp.CNNChroClass import CNN_ChroClass
import random, math, misc, warnings

class CNNGeneOperator:
    '''
    Chromosome Mutation Methods
    '''

    @classmethod
    def onep_crossover(self, parents:list):
        """
        Perform one point crossover onto a list of chromosome parents.

        Args:
            parents (list): A list of parents chromosomes

        Returns:
            child (list): A list of children chromosomes
        """
        self.__checker(parents)

        # scramble the parents, then crossover 1 by 1
        parents = random.sample(parents, len(parents))  # shuffle the parents
        child = list()
        if len(parents) < 2:
            raise Exception("Parents can't be less than 2!")

        if len(parents) % 2 != 0:  # if the list is not even
            warnings.warn("WARNING: Uneven parents. Discarding last member ..")
            del parents[-1]  # delete the last member

        for x in range(0, len(parents), 2):
            is_valid = False
            # turn chromosome into chunk for easier manipulation
            chunk_1, chunk_2 = CNN_ChroClass.chro_str2chunk(
                parents[x]), CNN_ChroClass.chro_str2chunk(parents[x+1])
            while is_valid == False:
                # get the index where f is started
                fin_chunk_1 = [i for i, s in enumerate(chunk_1) if 'F' in s][0]
                fin_chunk_2 = [i for i, s in enumerate(chunk_2) if 'F' in s][0]

                # now to swap them
                c_chunk_1 = chunk_1[:fin_chunk_1] + chunk_2[fin_chunk_2:]
                c_chunk_2 = chunk_2[:fin_chunk_2] + chunk_1[fin_chunk_1:]
                
                # turn chunk into string for checking validity
                chunk_1_str, chunk_2_str = CNN_ChroClass.chro_chunk2str(
                    c_chunk_1), CNN_ChroClass.chro_chunk2str(c_chunk_2)
                
                is_valid = True
                child.append(chunk_1_str)
                child.append(chunk_2_str)
        return child

    @classmethod
    def uniform_crossover(self, parents:list):
        """
        Perform uniform point crossover onto a list of chromosome parents.

        Args:
            parents (list): A list of parents chromosomes

        Returns:
            child (list): A list of children chromosomes
        """
        self.__checker(parents)

        # scramble the parents, then crossover 1 by 1
        parents = random.sample(parents, len(parents))  # shuffle the parents
        child = list()
        if len(parents) < 2:
            raise Exception("Parents can't be less than 2!")

        if len(parents) % 2 != 0:  # if the list is not even
            warnings.warn("WARNING: Uneven parents. Discarding last member ..")
            del parents[-1]  # delete the last member

        for x in range(0, len(parents), 2):
            is_valid = False
            # no need to set chunk back every times
            chunk_1, chunk_2 = CNN_ChroClass.chro_str2chunk(
                    parents[x]), CNN_ChroClass.chro_str2chunk(parents[x+1])
            while is_valid == False:
                # turn chromosome into chunk for easier manipulation

                # get the index where f is started
                fin_chunk_1 = [i for i, s in enumerate(chunk_1) if 'F' in s][0]
                fin_chunk_2 = [i for i, s in enumerate(chunk_2) if 'F' in s][0]

                # now to split convolution part with linear for uniform
                # if parents has unequal size, then stop swapping at the least
                # same as linear part too

                conv_stop = min(fin_chunk_1, fin_chunk_2)
                #conv_stop = fin_chunk_1 if fin_chunk_1 < fin_chunk_2 else fin_chunk_2
                conv_chunk_1,conv_chunk_2 = chunk_1[:fin_chunk_1],chunk_2[:fin_chunk_2]

                f_chunk_1,f_chunk_2 = chunk_1[fin_chunk_1:],chunk_2[fin_chunk_2:]
                # don't cross the last layer
                f_stop = (min(len(f_chunk_1),len(f_chunk_2))) - 2
                #f_stop = (len(f_chunk_1) if len(f_chunk_1) < len(f_chunk_2) else len(f_chunk_2)) - 2

                for x in range(0,conv_stop):
                    if random.randint(0,1) == 0: # swap
                        conv_chunk_1[x],conv_chunk_2[x] = conv_chunk_2[x],conv_chunk_1[x]
                    else: # do nothing
                        pass
                
                for x in range(0,f_stop):
                    if random.randint(0,1) == 0: # swap
                        f_chunk_1[x],f_chunk_2[x] = f_chunk_2[x],f_chunk_1[x]
                    else: # do nothing
                        pass

                r_chunk_1,r_chunk_2 = conv_chunk_1 + f_chunk_1,conv_chunk_2 + f_chunk_2
                
                # turn chunk into string for checking validity
                chunk_1_str, chunk_2_str = CNN_ChroClass.chro_chunk2str(
                    r_chunk_1), CNN_ChroClass.chro_chunk2str(r_chunk_2)
                
                #if CNN_ChroClass.chro_check(chunk_1_str) == -1 or CNN_ChroClass.chro_check(chunk_2_str) == -1:
                    #print("a chunk is not valid .. recrossing")
                    #continue # remove me later
                is_valid = True
                child.append(chunk_1_str)
                child.append(chunk_2_str)
        return child

    @classmethod
    def swap_crossover(self, parents: list):
        """
        (LEGACY)
        Perform swap crossover onto a list of chromosome parents.

        Args:
            parents (list): A list of parents chromosomes

        Returns:
            child (list): A list of children chromosomes

        Notes:
            This method isn't recommended due to heavy rely on checker method
        """
        self.__checker(parents)

        # scramble the parents, then crossover 1 by 1
        parents = random.sample(parents, len(parents))  # shuffle the parents
        child = list()
        if len(parents) < 2:
            raise Exception("Parents can't be less than 2!")

        if len(parents) % 2 != 0:  # if the list is not even
            warnings.warn("WARNING: Uneven parents. Discarding last member ..")
            del parents[-1]  # delete the last member

        for x in range(0, len(parents), 2):
            is_valid = False
            while is_valid == False:
                chunk_1, chunk_2 = CNN_ChroClass.chro_str2chunk(
                    parents[x]), CNN_ChroClass.chro_str2chunk(parents[x+1])
                random_select = (random.randint(0, len(chunk_1) - 2),
                                 random.randint(0, len(chunk_2) - 2))
                chunk_1[random_select[0]], chunk_2[random_select[1]
                                                   ] = chunk_2[random_select[1]], chunk_1[random_select[0]]
                # turn chunk into string for checking validity
                chunk_1_str, chunk_2_str = CNN_ChroClass.chro_chunk2str(
                    chunk_1), CNN_ChroClass.chro_chunk2str(chunk_2)
                if CNN_ChroClass.chro_check(chunk_1_str) == -1 or CNN_ChroClass.chro_check(chunk_2_str) == -1:
                    # print("a chunk is not valid .. recrossing")
                    continue
                is_valid = True
                child.append(chunk_1_str)
                child.append(chunk_2_str)
        return child

    @classmethod
    def random_mutation(self, parents: list, max_parents = None, max_kernel:int = 5, max_neurons:int = 100):
        """
        Perform random mutation onto a list of chromosome parents.

        Randomly replace, add, 'perform nothing', or remove a layer from a parents

        Args:
            parents (list): A list of parents chromosomes
            max_parents (int): Maximum number of parents to perform (default: None)
            max_kernel (int): Maximum number of kernel sizes to mutate (default: 5)
            max_neurons (int): Maximum number of neurons to mutate in linear layers (default: 100)

        Returns:
            child (list): A list of children chromosomes
        """
        
        if max_parents == None:
            max_parents = len(parents)

        if max_parents <= 0:
            raise Exception("Selected parents can't be less than zero!")
        elif max_parents >= 1 and max_parents <= 3:
            warnings.warn("Low numbers of parents for mutation are less likely to get the best result")

        # division the len of parents for fast whatever
        parents = parents[:int(max_parents)]
        self.__checker(parents)

        child = list()

        for x in range(0, len(parents)):
            is_valid = False
            while is_valid == False:
                chunk = CNN_ChroClass.chro_str2chunk(parents[x])
                random_select = random.randint(0, len(chunk) - 2)
                # do your stuff magic man
                choice = random.randint(0, 3)
                mutat = chunk[random_select][0] # select a random layer, then copy its type
                mutat = misc.random_layer(mutat)[0]

                if choice == 0:  # replace the position of layer
                    #print("replace")
                    chunk[random_select] = mutat
                elif choice == 1:  # remove a layer
                    #print("removal")
                    del chunk[random_select]
                elif choice == 2:  # add a layer
                    #print("addition")
                    chunk.insert(random_select, mutat)
                else:
                    pass # literally nothing

                chunk_str = CNN_ChroClass.chro_chunk2str(chunk)
                if CNN_ChroClass.chro_check(chunk_str) == -1:
                    # there's a problem
                    #print(chunk_str,"|",parents)
                    continue
                is_valid = True
                child.append(chunk_str)
        return child

    @classmethod
    def uniform_mutation(self, parents: list, max_parents = None, max_kernel:int = 5, max_neurons:int = 100):
        """
        Perform uniform mutation onto a list of chromosome parents.

        Scramble the parents, split to four subgroups. Then add, replace, remove, and 'do nothing' for all of them.

        Args:
            parents (list): A list of parents chromosomes
            max_parents (int): Maximum number of parents to perform (default: None)
            max_kernel (int): Maximum number of kernel sizes to mutate (default: 5)
            max_neurons (int): Maximum number of neurons to mutate in linear layers (default: 100)

        Returns:
            child (list): A list of children chromosomes
        """
        
        if max_parents == None:
            max_parents = len(parents)

        if max_parents <= 0:
            raise ValueError("Selected parents can't be less than zero!")
        elif max_parents >= 1 and max_parents < 4:
            raise ValueError("Can't perform uniform mutation with parents length less than 4!")

        # division the len of parents for fast whatever
        parents = parents[:int(max_parents)]
        parents = random.sample(parents, len(parents))  # shuffle the parents
        self.__checker(parents)

        a = math.floor(len(parents) / 4)
        seq = [a,(a*2),(a*3)] # threshold points

        child = list()

        for x in range(0, len(parents)):
            is_valid = False
            while is_valid == False:
                chunk = CNN_ChroClass.chro_str2chunk(parents[x])
                random_select = random.randint(0, len(chunk) - 2)
                # do your stuff magic man
                mutat = chunk[random_select][0] # select a random layer, then copy its type
                mutat = misc.random_layer(mutat)[0]

                # here comes the cursed if-else statement
                if x < seq[0]: # addition
                    #print("addition")
                    chunk.insert(random_select, mutat)
                elif x >= seq[0] and x < seq[1]: # replace
                    #print("replace")
                    chunk[random_select] = mutat
                elif x >= seq[1] and x < seq[2]: # removal
                    #print("removal")
                    del chunk[random_select]
                else: # do nothing and die
                    #print("sorry nothing")
                    pass

                chunk_str = CNN_ChroClass.chro_chunk2str(chunk)
                if CNN_ChroClass.chro_check(chunk_str) == -1:
                    #print(chunk_str,"|",parents)
                    continue
                is_valid = True
                child.append(chunk_str)
        return child

    def __checker(parents):
        '''
        (PRIVATE METHOD)

        Don't use this.
        Check if chromosomes are invalid or not while performing genetic operations
        '''
        for x in parents:
            if CNN_ChroClass.chro_check(x) == -1:
                raise Exception("Bad gene or invalid sequence:", x)