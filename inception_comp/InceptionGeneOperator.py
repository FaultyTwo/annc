from annc.inception_comp.InceptionChroClass import InceptionChroClass
import random, warnings, math, misc

# Generate proper comment, please

class InceptionGeneOperator:
    '''
    Gene operation for Inception chromosomes
    Subclasses: OuterCrossover, InnerCrossover, OuterMutation, InnerMutation
    '''
    # TODO: Check for all-different, all-same or n-same
    
    class OuterCrossover:
        '''
        Gene operations for outer scope
        '''
        @classmethod
        def outer_onep_crossover(self, parents:list):
            '''
            Perform one-point crossover to outer scope of a parents chromosome
            '''

            #self.__checker(parents)
            
            # scramble the parents, then crossover 1 by 1
            parents = random.sample(parents, len(parents))  # shuffle the parents

            child = list()
            if len(parents) < 2:
                raise Exception("Parents can't be less than 2!")
        
            if len(parents) % 2 != 0:
                warnings.warn("Parents can't be an even number! Removing last member ...")
                del parents[-1]
            
            for x in range(0, len(parents), 2):
                chunk_1 = InceptionChroClass.chro_str2chunk(parents[x]) # cover to chunk
                chunk_2 = InceptionChroClass.chro_str2chunk(parents[x+1]) # cover to chunk
                # get the lower length of the two chunks
                min_chunk = len(chunk_1) if len(chunk_1) < len(chunk_2) else len(chunk_2)
                cross = random.randint(1, min_chunk - 1) # we don't want to cross input
                res_1,res_2 = chunk_1[:cross] + chunk_2[cross:],chunk_2[:cross] + chunk_1[cross:]
                # no checker, flexible
                res_1,res_2 = InceptionChroClass.chro_chunk2str(res_1),InceptionChroClass.chro_chunk2str(res_2)
                child.append(res_1)
                child.append(res_2)
                
            return child
    
    class InnerCrossover:
        '''
        Gene operations for inner scope
        '''
        @classmethod
        def inner_onep_crossover(self, parents:list):
            '''
            Perform one-point crossover to inner scope of a parents chromosomes
            '''
            #self.__checker(parents)

            parents = random.sample(parents, len(parents))  # shuffle the parents

            child = list()
            if len(parents) < 2:
                raise Exception("Parents can't be less than 2!")
        
            if len(parents) % 2 != 0:
                warnings.warn("Parents can't be an even number! Removing last member ...")
                del parents[-1]

            for x in range(0, len(parents), 2):
                chunk_1 = InceptionChroClass.chro_str2chunk(parents[x]) # cover to chunk
                chunk_2 = InceptionChroClass.chro_str2chunk(parents[x+1]) # cover to chunk
                min_chunk = min(len(chunk_1),len(chunk_2)) # get lower length
                # in inner crossover, we need to perform one_p crossover for each members inside an inception
                # and due to that .. shit
                for y in range(0, min_chunk):
                    if type(chunk_1[y]) == str or type(chunk_2[y]) == str:
                        # do nothing since they are just a basic cnn
                        pass
                    else:
                        # perform inner crossover
                        min_len = min(len(chunk_1[y]), len(chunk_2[y]))
                        onep = random.randint(0, min_len - 1)
                        chunk_1[y],chunk_2[y] = chunk_1[y][:onep] + chunk_2[y][onep:],chunk_2[y][:onep] + chunk_1[y][onep:]

                res_1,res_2 = InceptionChroClass.chro_chunk2str(chunk_1),InceptionChroClass.chro_chunk2str(chunk_2)
                child.append(res_1)
                child.append(res_2)
            
            return child
    
        @classmethod
        def inner_uniform_crossover(self, parents:list):
            '''
            Perform uniform crossover to inner scope of parents chromosomes
            '''
            #self.__checker(parents)

            parents = random.sample(parents, len(parents))  # shuffle the parents

            child = list()
            if len(parents) < 2:
                raise Exception("Parents can't be less than 2!")
        
            if len(parents) % 2 != 0:
                warnings.warn("Parents can't be an even number! Removing last member ...")
                del parents[-1]

            for x in range(0, len(parents), 2): # handle each chromosomes
                chunk_1 = InceptionChroClass.chro_str2chunk(parents[x]) # cover to chunk
                chunk_2 = InceptionChroClass.chro_str2chunk(parents[x+1]) # cover to chunk
                min_chunk = min(len(chunk_1),len(chunk_2)) # get lower length
                # in inner crossover, we need to perform one_p crossover for each members inside an inception
                # and due to that .. shit
                for y in range(0, min_chunk): # handle each chunks
                    if type(chunk_1[y]) == str or type(chunk_2[y]) == str:
                        # do nothing since they are just a basic cnn
                        continue
                    else:
                        # perform inner crossover
                        min_len = min(len(chunk_1[y]), len(chunk_2[y]))

                        for z in range(0,min_len): # handles each layer inside an inception
                            if bool(random.randint(0,1)):
                                chunk_1[y][z],chunk_2[y][z] = chunk_1[y][z],chunk_2[y][z] # swap the layer

                res_1,res_2 = InceptionChroClass.chro_chunk2str(chunk_1),InceptionChroClass.chro_chunk2str(chunk_2)
                child.append(res_1)
                child.append(res_2)
            
            return child
    
    class OuterMutation:

        @classmethod
        def outer_mutation(self, parents:list):
            '''
            Perform one-point mutation to outer scope of parents chromosomes
            '''
            #self.__checker(parents)

            # implement max parents later
            valid_layers = ['C','P']
            valid_out_chan = ['8','16','32','64']

            child = list()
            
            for x in range(0, len(parents)):
                chunk = InceptionChroClass.chro_str2chunk(parents[x])
                new_block = [] # empty the new_block
                # we have chunks. now what?
                # first, we generate a new inception block first.
                # get the largest inception block size => ceil(size/2) => generate a new inception block
                # using that size

                pos = random.randint(1, len(chunk) - 2)
                choice = random.randint(0, 2)

                if choice == 0:
                    #print("NOTHING! NOW GET OUT!")
                    pass # do nothing
                elif choice == 1: # removal
                    #print("Deleted {}".format(pos))
                    del chunk[pos]
                else: # addition + replacement
                    # get average size from all inception blocks
                    max_size = math.ceil((max([len(ch) for ch in chunk if type(ch) != str])) / 2)
                    for _ in range(0,max_size):
                        new_layer = misc.random_layer(random.choice(valid_layers))[0]
                        if new_layer[0] == 'C': # add output channel for convolution
                            new_layer += random.choice(valid_out_chan)
                        new_block.append(new_layer)
                    if bool(random.randint(0,1)): # if false, addition, else, replacement
                        #print("Replacement {}".format(pos))
                        chunk[pos] = new_block
                    else:
                        #print("Added {}".format(pos))
                        chunk.insert(pos, new_block)
                #print(chunk)
                res = InceptionChroClass.chro_chunk2str(chunk)
                child.append(res)

            return child
        
# excuse me but what's the command for commit a git?
# git add .
# git commit -m "commit"
# git push origin master
# ok thanks