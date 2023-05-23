import numpy as np
import pandas as pd

class Prepare_data :
    """
    this class is to add padding to our data, extract the vocabulary, and create out training batches
    """

    def __init__(self, data_path):

        self.data_path = data_path
        # we extract the data into a dataframe
        self.data = pd.read_csv(self.data_path)

        # our padding character : 
        self.pad_char = '%'

        # our final padded data ready for training + indexed dictionnary for our char emebeddings
        self.final_data, self.char_index_dic = self.set_data()  



    def set_data(self):
        '''
        this function is to calculate the max length of our words and roots and padd it to the same length
        '''
        words,roots = self.data['word'].values.tolist(),self.data['root'].values.tolist()
        data = self.data.values.tolist() 

        # Let's calculate the biggest length
        max_len_words = max([len(item) for item in words])
        max_len_roots = max([len(item) for item in roots])   

        padded_data = []
        vocab = []

        for instance in data: 
            tmp = []
            word,root = instance[0], instance[1]
            while(len(word) != max_len_words):
                word += self.pad_char
            tmp.append(word)
            while(len(root) != max_len_roots):
                root += self.pad_char
            
            instance_set = set(word+root)

            for item in instance_set :
                if item not in vocab :
                    vocab.append(item)
            
            tmp.append(root)
            padded_data.append(tmp)
            

        # now we can create our indexed dictionnary
        char_to_idx_map = {char: idx for idx, char in enumerate(vocab)}

        return padded_data, char_to_idx_map






