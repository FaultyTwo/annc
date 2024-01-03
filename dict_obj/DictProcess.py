from annc.res_comp.ResChroClass import ResChroClass
from annc.inception_comp.InceptionChroClass import InceptionChroClass
from annc.cnn_comp.CNNChroClass import CNN_ChroClass
import json, random, string

# This is becoming a carbon copy of amoebanet dictionary
# Ok. So your prof. wants you to implement a dict object then turn them into models or chromosomes
# Let us remind what you think would work

'''
{
    "1": "C22",
    "2": "P22",
    "3": "ISC3316SC2216I",
    "4": "R1"
    "5": "C33P22"
    "6": "R1",
    "7": "F10"
}
Turning this into network would give us, "C22P22ISC3316SC2216IR1C33P22R1F10"
Networks are build from the first key to the last key
IF we are going to use this, we need to create the model_build function that supports every types of chromosomes
Oh puke. He should'd told me this before I implement each of them seperately.
*shrugs*

Well. This dict is going to get more wild
Reference:
- Cut out 'R' alphabet since we don't need them anymore due to we are going to create another dict containing
the information about connections from each keys instead of relying on 'R'.
Ex.
{
    "1": "C22",
    "2": "P22",
    "3": "ISC3316SC2216I",
    "4": "C33P22"
    "5": "F10"
}
connection:
{
    "1": "", # either leave as "0" or null, since first node input is.. the data
    "2": "1",
    "3": "1","2" # residual in work
    "4": "C33P22"
    "5": "F10"
}
valid connection:
T = current key
x = random(x) where x in range of [1,T]
valid = random(x)
(for random generations)
'''

class DictProcess:

    @classmethod
    def dict_to_chromosome(self, gene_dict:dict, return_type:str = "str"):
        """
        Converts a gene dictionary into a chromosome.

        Args:
            gene_dict (dict): A dictionary containing the gene information.
            return_type (string): Return the chromosome in string or chunk format
                "str": return the chromosome in a string
                "chunk": return the chromosome in a list

        Returns:
            str: A chromosome string representing the gene.

        Raises:
            Exception: If the 'con' key is missing in the gene dictionary.

        Notes:
            This function sorts the dictionary based on the key value, appends the 'R' connection to the dict 
            first, and converts the dictionary to a chromosome string.
        """

        # I have an awful idea to solve this stupid residual keying problem
        # since our input is a dict, maybe we should append the 'R' connection to the dict first
        # then, we sort it based on key value
        # jesus christ on a laptop, it works.
        conn = gene_dict.pop('con', None)
        if conn == None:
            raise Exception('Missing \'con\' key in dict')
        
        uh = 0.001 # i just found out you can use float for.. key in the dict, wtf?
        # i know using decimal for a key is a bad idea
        # in the future, i might use string instead
        # but for now. eh. it's probably for the best.
        residual_id = 1
        dummy = dict()
        for k,v in conn.items():
            if type(v) is int or v == None:
                continue
            else:
                if len(v) == 1:
                    continue # normal forwarding
                else: # remember. every connection starts after the layer
                    for y in v:
                        gene_dict[y + uh] = 'R' + str(residual_id)
                    residual_id += 1
                    uh += 0.001
        
        gene_dict = dict(sorted(gene_dict.items())) # this is cursed
        #print(gene_dict)

        if return_type == "str":
            cool_chro = str()
            for k,v in gene_dict.items():
                if type(v) is list:
                    cool_chro += InceptionChroClass.chro_chunk2str([v])
                else:
                    cool_chro += v
        elif return_type == "chunk":
            cool_chro = list()
            for k,v in gene_dict.items():
                cool_chro.append(v)
        else:
            raise Exception("Unknown return types, must be \"chunk\" or \"str\"")
        
        return cool_chro
        '''
        for key, value in gene_dict.items():
            if type(key) is int:
                if type(value) is list:
                    temp = [value] # rubber band solution
                    list_str.append(InceptionChroClass.chro_chunk2str(temp))
                else:
                    list_str.append(value)
            else:
                if key == 'con':
                    break # break the loop at last, we won't allow any further keys
                else:
                    raise Exception('Invalid key', str(key))
        '''
    
    @classmethod
    def dict_generate(self, population:int, length_range:set, output_channel:int):
        # either skip this method entirely since you could just use chromosomes instead
        # or not.. who knows?
        raise NotImplementedError("This method is not implemented yet")
    
    @classmethod
    def chromosome_to_dict(self, gene:str, write_to_file=False, path = None):
        """
        Generates a dictionary from a chromosome string.

        Args:
        - gene (str): The chromosome string to be converted into a dictionary
        - write_to_file (bool): Whether or not the resulting dictionary should be saved to a JSON file
        - path: A directory path to save a JSON file

        Returns:
        - dict: The dictionary generated from the chromosome string
        """ 

        chunk = ResChroClass.chro_str2chunk(gene) 
        # seperate residual connection first
        # then, seperate inception later

        # non-residual path network seperator
        new_chunk = list()
        r_chunk = list()
        for x in chunk:
            #print(x)
            if x[0] == 'R':
                r_chunk.append(x)
                continue
            temp = InceptionChroClass.chro_str2chunk(x)
            for y in temp:
                if type(y) is list:
                    new_chunk.append(y)
                    r_chunk.append('N')
                else:
                    temp_2 = CNN_ChroClass.chro_str2chunk(y)
                    for z in temp_2:
                        new_chunk.append(z)
                        r_chunk.append('N')
        #print(new_chunk)
        #print(r_chunk)

        dict_obj = dict() # returns the chunks as a dict

        for idx,x in enumerate(new_chunk):
            # TODO: Write a validity check for each member inside each key
            # Actually don't. Waste of CPU cycle.
            dict_obj[idx+1] = x
        
        # how am i going to represent the residual connections from the chunk?
        # gotta plan this carefully .. there's always a trick to do this
        # and i need to implement it perfectly
        conn_obj = dict() # returns connections as a dict .. woo we
        res_obj = dict() # checking if a residual point has started or not, or checking where connections are come from
        penal = 0
        prev_r = False

        for idy,y in enumerate(r_chunk):
            #print(idy, ":")
            if idy == 0:
                conn_obj[idy+1] = None
                continue
            
            if y != 'N': # if there's a residual connection
                if y not in res_obj:
                    res_obj[y] = [idy - penal] # create a list
                else:
                    res_obj[y].append(idy - penal) # append a residual position
                #print(res_obj)
                penal += 1 # made up for missing connection in main network
                prev_r = True
                prev_lay = y
            else:
                if prev_r:
                    # STOP, REFERENCING, TO, THE, OBJECT! AHHHHHHHHHH!
                    # Copy the residual connection (list) to this current layer
                    conn_obj[idy+1-penal] = res_obj[prev_lay].copy()
                    prev_r = False # not a prev_r anymore
                else:
                    conn_obj[idy+1-penal] = idy - penal # no residual detected
            #print(conn_obj)

        dict_obj['con'] = conn_obj

        if write_to_file:
            random_str = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(12))
            # note: above line is the way to use loop argument without variable .. wow!
            with open(random_str + '.json', 'w') as f:
                json.dump(dict_obj, f, indent=4)

        return dict_obj

