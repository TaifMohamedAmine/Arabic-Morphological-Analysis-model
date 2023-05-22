from main import Model
from Preprocess import Prepare_data


data_inst = Prepare_data("projet NLP/data.csv", batch_size=40)
batches, char_idx = data_inst.batches, data_inst.char_index_dic
print("*"*10+' BATCHES READY '+"*"*10)
print("\n")

model = Model(batches, char_idx, embedding_size=128, hidden_size=40,
                num_layers=3, teacher_forcing_ratio=0.35, dropout=0.2,
                    learning_rate=0.005, training_ratio=0.8)
print("*"*10+' MODEL INITIATED '+"*"*10)
print("\n")

print("the number of our models parameters : ", model.count_parameters(),'parameters')

print("\n")

print("*"*10+' Lets train our model !!!! '+"*"*10)
print("\n")


losses = model.fit(num_epochs=2)


