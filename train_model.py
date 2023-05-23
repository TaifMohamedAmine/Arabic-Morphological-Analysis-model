from main import Model
from Preprocess import Prepare_data
import matplotlib.pyplot as plt
import numpy as np


data_inst = Prepare_data("projet NLP/data.csv")
data, char_idx = data_inst.final_data, data_inst.char_index_dic
print("the number of samples in our data: ", len(data))
print("*"*10+' BATCHES READY '+"*"*10)
model = Model(data, char_idx, batch_size=500, embedding_size=100, hidden_size=40,
                num_layers=1, teacher_forcing_ratio=0.35, dropout=0.3,
                    learning_rate=0.001, training_ratio=0.8)
print("*"*10+' MODEL INITIATED '+"*"*10)
print("the number of our models parameters : ", model.count_parameters(),'parameters')
print("*"*10+' Lets train our model !!!! '+"*"*10)

num_epochs = 120

losses = model.fit(num_epochs)

losses = np.array(losses)

plt.figure()
plt.plot(list(range(num_epochs)), losses[:, 0], c = 'r')
plt.plot(list(range(num_epochs)), losses[:, -1], c = 'b')
plt.show()

