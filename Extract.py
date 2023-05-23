import pandas as pd
from xml.etree import ElementTree as et 
import pyarabic.araby as araby
import numpy as np


'''
this file is for text extractiion and preprocessing to a csv file  
'''


class Preprocessing :
    '''
    in this class we will be preparing our data for training our Arabic morphological analyzer
    '''
    
    def __init__(self,xml_path):
        self.file_path = xml_path
        
        # an empty list for our extracted morphems 
        self.text = []

        # we define the start and end chars of our data 
        self.sow = '$' # start-of-word
        self.eow = '£' # end-of-word




    def extract_text(self):
        '''
        this function is to extract text morphems from the xml file
        '''
        tree = et.parse(self.file_path)
        xml_root=tree.getroot()

        # we initiate with an empty data list 
        data = []

        num_file = 0
        for file in xml_root.findall("FILE"):
            print("extracting text from file ", num_file)
            num_file += 1
            for sentence in file.findall("sentence"):
                annotation = sentence.find("annotation")
                for morphem in annotation.findall("ArabicLexical"):
                    # eliminate the words with no morphems
                    if (morphem.get("pos")!= 'غير عربية') and (len(araby.strip_diacritics(morphem.get('lemma')))>= 3): 
                        # remove punctuation from the word
                        word = self.sow + araby.strip_diacritics(morphem.get('lemma')) + self.eow
                        root = self.sow + araby.strip_diacritics(morphem.get('root')) + self.eow
                        data.append([word, root])
        return data

    def Preprocess(self):
        '''
        this function is to call the extracted text
        '''
        text = np.array(self.extract_text())
        cols = ['word', 'root']
        data = pd.DataFrame(text, columns=cols)
        print(data)
        data.to_csv("projet NLP/data.csv", index=False)

        return text

if __name__ == '__main__': 
    
    inst = Preprocessing("projet NLP/Nemlar/NEMLAR.xml")
    text = inst.Preprocess()

    
















