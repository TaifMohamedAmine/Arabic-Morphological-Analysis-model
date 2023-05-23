import torch 
import torch.nn as nn
import numpy as np
from main import Model
from Preprocess import Prepare_data


data_inst = Prepare_data("projet NLP/data.csv")
data, char_idx = data_inst.final_data, data_inst.char_index_dic
print("the number of samples in our data: ", len(data))
print("*"*10+' BATCHES READY '+"*"*10)
loaded_model = Model(data, char_idx, batch_size=300, embedding_size=100, hidden_size=40,
                num_layers=1, teacher_forcing_ratio=0.35, dropout=0.3,
                    learning_rate=0.0001, training_ratio=0.8)


loaded_model.load_state_dict(torch.load("projet NLP/best_model.pt"))
test_word = '$' + 'تحليل' + '£'
print(loaded_model.predict(test_word))














