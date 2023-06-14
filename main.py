import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import random 
from Preprocess import Prepare_data
import time
from sklearn.preprocessing import StandardScaler


"""
I need to add positionnal embeddings for my encoder ==> done
"""


class Model(nn.Module) :
    '''
    Our model training class :p
    '''
    def __init__(self, data, char_idx, batch_size,  embedding_size, hidden_size, num_layers, teacher_forcing_ratio, dropout = 0.2, learning_rate = 0.001, training_ratio = 0.8):
        super().__init__()
        '''
        first we define our models parameters
        '''
        # our data delimiters :
        self.sow = '$'
        self.eow = '£'

        # beam search decoder global list and counter 
        self.counter = 0
        self.result_list = []

        # our optimizer learning rate :
        self.lr = learning_rate

        # our training data : 
        self.data = data

        # our batches :
        self.batches = None
        self.batch_size = batch_size 

        # our training ratio : 
        self.ratio = 0.8

        # the padding char used :
        self.pad_char = '%'

        # our char index dictionnary : 
        self.char_index_dic = char_idx

        # embedding size and hidden size : 
        self.embedding_size , self.hidden_size = embedding_size, hidden_size

        # the dropout ratio for our layers : 
        self.dropout = dropout

        # our training and validation batches 
        self.train_batches, self.val_batches = self.prepare_batches()

        '''
        Now Let's define our Neural Network structure 
        '''

        # Our teacher forcing ratio :
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Our embedding layer : 
        self.embedding = nn.Embedding(num_embeddings = len(self.char_index_dic), embedding_dim = self.embedding_size, padding_idx = self.char_index_dic[self.pad_char])

        # Our dropout layer : 
        self.Dropout = nn.Dropout(self.dropout)

        # *********************Our layers**************************** : 

        # number of layers : 
        self.num_layers = num_layers
        
        # the encoder layer
        self.BILSTM = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True, dropout = self.dropout)
        self.input_dense = nn.Linear(self.hidden_size * 2,self.embedding_size)

        # the decoder layer
        self.LSTM = nn.LSTM(input_size= self.embedding_size ,hidden_size = self.hidden_size*2, num_layers = self.num_layers, batch_first = True , dropout = self.dropout)

        # the output dense layer
        self.Linear = nn.Linear(self.hidden_size * 2,len(self.char_index_dic))

        # our loss function : 
        self.criterion = nn.CrossEntropyLoss(ignore_index =self.char_index_dic[self.pad_char])

        # Our optimizers used : 
        self.opt1 = optim.Adam([*self.embedding.parameters(), *self.BILSTM.parameters()], lr = self.lr )
        self.opt2 = optim.Adam([*self.LSTM.parameters(), *self.input_dense.parameters(), *self.Linear.parameters()], lr = self.lr)

        print("all the parameters are initialized")

    

    def word_to_seq(self, word):
        '''
        this function returns a sequence of the unique indexes for the given word 
        (sequence is tensor that can be changed using a .tolist() )
        '''
        word_char_idx_seq =[self.char_index_dic[char] for char in word]    
        return word_char_idx_seq # word sequence
    

    def positional_encoding(self, embedded_batch):
        """
        this method is to add a positionnal encoding for the a given embedded input sequence
        
        input : 
            - embedded input batch 
        
        returns : 
            - embedded sequence with positionnal encoding     
        """

        # the shape of the embedded vec (batch size, sequence length, embedding size)
        pos_embedding = torch.zeros(embedded_batch.size())
        #print(pos_embedding.size())

        for idx in range(embedded_batch.size(0)):
            for char in range(embedded_batch.size(1)):
                for pos in range(embedded_batch.size(2)) :
                    if pos % 2 : # when the position index is odd
                        pos_embedding[idx][char][pos] = np.cos(char / (10000 **(2 * pos / embedded_batch.size(2))))
                    else : 
                        pos_embedding[idx][char][pos] = np.sin(char / (10000 **(2 * pos / embedded_batch.size(2))))


        return pos_embedding




    # Let's now build out model :)

    def encode(self, batch, pos_embedding):
        '''
        input : a batch of sequences of instances : [word_seq , root_seq] * batch_size
                input_size : (input_size,2)
        '''
        word_batch = [] # list of words in the batch
        root_batch = [] # list of roots in the batch
        for instance in batch : 
            word_batch.append(instance[0])
            root_batch.append(instance[1])
        word_batch = torch.tensor(word_batch)
        root_batch = torch.tensor(root_batch)
        
        # we create embedding of the word batch : 
        embedded_word_batch = self.embedding(word_batch) + pos_embedding 
 
        # we initialize the weights of the encoder network with a normal distribution
        init_hid = nn.init.xavier_normal_(torch.zeros(2*self.num_layers, len(batch), self.hidden_size), gain=0.5)
        init_ce = nn.init.xavier_normal_(torch.zeros(2*self.num_layers, len(batch), self.hidden_size), gain=0.5)
            
        outputs, (hidden, cell) = self.BILSTM(embedded_word_batch, (init_hid, init_ce)) # we pass the emebedded vector through the bi-GRU 
        
        '''
        hidden size : [2 * num_layers, batch_size , hidden_size]        
        we want hidden size : [num_layers , batch_size  , 2 * hidden_size]
        we return an adequate layer for the decoder : 
        '''

        final_hid, final_ce = [], []
        for k in range(0,hidden.size(0), 2):
            
            tmp_hid = hidden[k:k+2 , :, :]
            tmp_ce = cell[k:k+2, :, :]
        
            cct_hid = torch.cat((tmp_hid[0], tmp_hid[1]), dim  = 1).tolist()
            cct_ce = torch.cat((tmp_ce[0], tmp_ce[1]), dim  = 1).tolist()
            
            final_hid.append(cct_hid)
            final_ce.append(cct_ce)
        
        final_hid, final_ce = torch.tensor(final_hid), torch.tensor(final_ce)
        return root_batch , outputs ,(final_hid, final_ce)
    


    def decode(self, encoder_hidden_cell , batch, teacher_forcing_bool, epoch):
        '''
        input : encoding_hidden_layer => corresponds to the concatenation of the final hidden layers 
                                        of the bidirectionnal lstm in our encoder
                
                batch : subset of data that contains the roots of the words we encoded.
                
        output : we'll see :) 
        
        '''

        (hidden_layer , cell) , root_batch = encoder_hidden_cell , batch 
        embedded_char = self.embedding(torch.unsqueeze(root_batch[:, 0], 1))
        outputs = []
        #topk_indexes = []
        for i in range(root_batch.size(1)): 
            self.Dropout(embedded_char)
            decoder_output , (hidden_layer, cell) = self.LSTM(embedded_char, (hidden_layer, cell))
            # Let's calculate the scores  :
            input_decoder_output = self.input_dense(decoder_output)
            embedded_char = input_decoder_output
            mask = np.where([random.random() <= (self.teacher_forcing_ratio) for i in range(root_batch.size(0))])[0]
            teacher_forcing_input = self.embedding(torch.unsqueeze(torch.clone(root_batch[:, i]), 1))
            if teacher_forcing_bool : 
                embedded_char[mask] = teacher_forcing_input[mask] 
            Dense_decoded_output = self.Linear(decoder_output)
            soft = nn.Softmax(dim = 2)
            soft_out = soft(Dense_decoded_output)
            outputs.append(soft_out)
        return outputs
    
        

    def train_model(self, batches, teacher_forcing_bool, epoch):
        '''
        training function 
        '''
        train_batches = batches           
        epoch_loss = 0
        n = 0            
        test_word = '$' + 'تحليل' + '£'

        pos_encoding = self.positional_encoding(self.embedding(torch.tensor([item[0] for item in train_batches[0]])))

        for batch in train_batches :
            #print(self.predict(test_word))
            self.opt1.zero_grad()
            self.opt2.zero_grad()
            if len(batch) == self.batch_size :
                root_batch, encoder_output, encoder_states = self.encode(batch, pos_encoding)
            else : 
                root_batch, encoder_output, encoder_states = self.encode(batch, self.positional_encoding(self.embedding(torch.tensor([item[0] for item in batch]))))
            outputs = self.decode(encoder_states, root_batch, teacher_forcing_bool, epoch)
            a = [torch.squeeze(item, 1) for item in outputs]
            a = [torch.unsqueeze(item, 0) for item in a]
            output = torch.cat(a, dim = 0)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = root_batch.transpose(0, 1)
            trg = trg.reshape(-1)

            """print(output, trg)
            raise Exception("")"""
            loss = self.criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*self.LSTM.parameters(), *self.BILSTM.parameters()], 1)
            self.opt1.step()
            self.opt2.step()       
            #self.optimizer.step()
            epoch_loss+=loss.item()
            n+=1
            print('the loss of the train batch ', n ,' is : ', loss.item())
    
        return epoch_loss/n
    
    def evaluate_model(self, batches, teacher_forcing_bool, epoch):
        '''
        this method evaluates our model :=)
        will be similar to train but without the teacher forcing/ using an optimizer 
        '''          
        self.eval()
        val_batches = batches
        n = 0
        epoch_loss = 0
        pos_encoding = self.positional_encoding(self.embedding(torch.tensor([item[0] for item in val_batches[0]])))
        with torch.no_grad() :
            for batch in val_batches :
                if len(batch) == self.batch_size :
                    root_batch, encoder_output, encoder_states = self.encode(batch, pos_encoding)
                else : 
                    root_batch, encoder_output, encoder_states = self.encode(batch, self.positional_encoding(self.embedding(torch.tensor([item[0] for item in batch]))))
                outputs = self.decode(encoder_states, root_batch, teacher_forcing_bool, epoch)
                a = [torch.squeeze(item, 1) for item in outputs]
                a = [torch.unsqueeze(item, 0) for item in a]
                output = torch.cat(a, dim = 0)
                output_dim = output.shape[-1]
                output = output.view(-1, output_dim)
                trg = root_batch.transpose(0, 1)
                trg = trg.reshape(-1)
                #print(output.size(), trg.size())
                loss = self.criterion(output, trg)
                epoch_loss+=loss.item()
                n+=1
                print('the loss of the val batch ', n ,' is : ', loss.item())

        return epoch_loss / n
    
    

    def each_char(self,depth, acc_char, predicted_chars, predicted_proba, hidden, cell):

        #print("the parms :", depth, result_list, acc_char, predicted_chars, predicted_proba)

        acc1, acc2, acc3 = acc_char[:], acc_char[:], acc_char[:] 

        test_acc_char = acc_char
        #print("hh", test_acc_char)

        char1, prob1 = predicted_chars[0],predicted_proba[0]
        char2, prob2 = predicted_chars[1],predicted_proba[1]
        char3, prob3 = predicted_chars[2],predicted_proba[2]

        acc_char1,predicted_chars1, predicted_proba1, hidden1, cell1 = self.run_char_through_NN(acc1, char1, prob1, hidden, cell, depth)
        acc_char2,predicted_chars2, predicted_proba2, hidden2, cell2 = self.run_char_through_NN(acc2,char2, prob2, hidden, cell, depth)
        acc_char3,predicted_chars3, predicted_proba3, hidden3, cell3 = self.run_char_through_NN(acc3, char3, prob3, hidden, cell, depth)

        #print(acc_char1, acc_char2, acc_char3)

        test_acc_char = acc_char1
        
        if len(test_acc_char) >= 5 :
            #print(test_acc_char)
            self.result_list.append([acc_char1,predicted_proba1])
            self.result_list.append([acc_char2,predicted_proba2])
            self.result_list.append([acc_char3,predicted_proba3])
            self.counter += 3
            #print('stopping cond incremented +3 :', self.counter)
            d = depth ** 4
            #print(self.counter, d)
            if self.counter >= d: return 
            

        else : 

            self.each_char(depth,acc_char1,predicted_chars1, predicted_proba1, hidden1, cell1)
            self.each_char(depth,acc_char2,predicted_chars2, predicted_proba2, hidden2, cell2)
            self.each_char(depth,acc_char3,predicted_chars3, predicted_proba3, hidden3, cell3)



                

    def run_char_through_NN(self,acc_char, char, proba, hidden, cell, depth): 
        acc = acc_char
        acc.append(char)
        #print("haizueazube", acc, char)
        if len(acc_char) == 5:
            return acc,char, proba, hidden , cell
        else :  
            # input :  item , item , item , item 
            embedded_char = torch.unsqueeze(self.embedding(torch.tensor(char)), 0)
            # we create a softmax layer : 
            soft = nn.Softmax(dim = 1)
            
            #print(hidden.size(), cell.size())
            decoder_output , (final_hidden, final_cell) = self.LSTM(embedded_char, (hidden, cell))
            input_decoder_output = self.input_dense(decoder_output)
            embedded_char = input_decoder_output
            Dense_decoded_output = self.Linear(decoder_output)
            output_proba = soft(Dense_decoded_output)
            best_proba, best_indices = torch.topk(output_proba, depth).values.tolist()[0], torch.topk(output_proba, depth).indices.tolist()[0]

            predicted_chars = best_indices
            predicted_proba = [proba*item for item in best_proba] # accumulated probability

            # returns list , list, item , item
            return acc,predicted_chars, predicted_proba, final_hidden, final_cell


    
    def beam_search_decoding(self, hidden, cell , depth):

        init_char = self.char_index_dic[self.sow]

        acc_char,predicted_chars, predicted_proba, final_hidden, final_cell = self.run_char_through_NN([], init_char, 1, hidden , cell, depth)
        
        #print("initial parms : ",acc_char,predicted_chars,predicted_proba)

        self.each_char(depth,acc_char,predicted_chars, predicted_proba, final_hidden, final_cell)

        res = self.result_list

        sorted_list = sorted(res, key = lambda x : x[1], reverse=True) 

        selected_char = [item[0][1:-1] for item in sorted_list[0:10]]

        print('the most probable outputs using beam search decoding are :', selected_char)

        return selected_char



    def predict(self, word):

        test_word = self.sow + word + self.eow
        # Let's turn the word into a sequence of word indexes 
        word_seq = self.word_to_seq(test_word)
        # Let's create an embedding of the word seq
        embedded_word = self.embedding(torch.tensor(word_seq))
        init_hid = nn.init.xavier_normal_(torch.zeros(2*self.num_layers, self.hidden_size), gain=0.5)
        init_ce = nn.init.xavier_normal_(torch.zeros(2*self.num_layers, self.hidden_size), gain=0.5)
        # Let's feed our word embedding to the encoder network
        outputs, (hidden, cell) = self.BILSTM(embedded_word, (init_hid, init_ce))
        
        #print(hidden.size())
        final_hid, final_ce = [], []
        for k in range(0,hidden.size(0), 2):
            
            tmp_hid = hidden[k:k+2 ,:]
            tmp_ce = cell[k:k+2, :]

            cct_hid = torch.cat((tmp_hid[0], tmp_hid[1]), dim  = -1).tolist()
            cct_ce = torch.cat((tmp_ce[0], tmp_ce[1]), dim  = -1).tolist()

            final_hid.append(cct_hid)
            final_ce.append(cct_ce)
    
        final_hidden, final_cell = torch.tensor(final_hid), torch.tensor(final_ce)

        results = self.beam_search_decoding(final_hidden, final_cell, depth=3)

        key_list = list(self.char_index_dic.keys())
        val_list = list(self.char_index_dic.values())

        final_roots =[]
        for seq in results : 
            position = [val_list.index(item) for item in seq]
            result_char = [key_list[pos] for pos in position]
            predicted_root = ''.join(result_char)
            final_roots.append(predicted_root)

        return final_roots     

    def data_2_seq(self): 
        '''
        this function indexes our text data
        '''
        final_data = []
        for instance in self.data : 
            tmp = []
            word = [self.char_index_dic[char] for char in instance[0]]
            root = [self.char_index_dic[char] for char in instance[1]]
            tmp.append(word)
            tmp.append(root)
            final_data.append(tmp)
        return final_data
        


    def data_batches(self, data):
        '''
        this function creates our training batches
        '''
        size= self.batch_size 
        batches = [data[i:i + size] for i in range(0, len(data), size)]
        return batches
    
    def scale(self ,train_data , val_data):
        '''
        scale our data with standard scaler
        '''
        std_word, std_root =  StandardScaler(), StandardScaler()
        
        train_word, train_root,val_word, val_root = [],[], [], []
        for instance in train_data: 
            train_word.append(instance[0])
            train_root.append(instance[1])
        for instance in val_data : 
            val_word.append(instance[0])
            val_root.append(instance[1])
        
        std_word.fit_transform(train_word)
        std_word.transform(val_word)
        std_root.fit_transform(train_root)
        std_root.transform(val_root)

        train_data = [list(item) for item in zip(train_word, train_root)]
        val_data = [list(item) for item in zip(val_word, val_root)]


        return train_data, val_data

    def prepare_batches(self):
        data = self.data_2_seq()
        data = random.sample(data, len(data))
        data_size = len(data)
        middle_index = int(data_size * self.ratio)
        train_data, val_data = data[:middle_index], data[middle_index:]
        #train_data , val_data = self.scale(train_data , val_data)
        train_batches, val_batches = self.data_batches(train_data), self.data_batches(val_data)
        print("the number of training batches: ", len(train_batches))
        print("the number of validation batches: ", len(val_batches))
        return train_batches, val_batches
    

    def fit(self, num_epochs):
        """
        let's first prepare our data      
        """
        print(f'The model has {self.count_parameters():,} trainable parameters')

        epochs = list(range(num_epochs))
        best_val_loss = 99999
        best_model_par = 0
        losses =[]
        test_word = '$' + 'تحليل' + '£'
        for epoch in epochs : 
            print('epoch num : ', epoch) 
            t1 = time.time()
            train_batches = random.sample(self.train_batches , len(self.train_batches))
            train_loss= self.train_model(train_batches, 1, epoch)
            val_loss = self.evaluate_model(self.val_batches, 0, epoch) # we set the teacher forcing to false            
            t2 = time.time()
            
            tmp = [train_loss, val_loss]
            losses.append(tmp)
            
            print('the training loss : ', train_loss , 'the val loss :', val_loss)
            print('epoch num : ' ,epoch , ' lasted : ', t2 - t1 , 'seconds')
            
            if val_loss < best_val_loss :
                best_val_loss = val_loss 
                best_model_par = self.state_dict()
        torch.save(best_model_par, 'projet NLP/best_model3.pt')
        predicted_root = self.predict(test_word)
        print(predicted_root)       
        return losses
    
    def count_parameters(self):
        '''
        function to calculate the total number of parameters in the model
        '''
        return sum(torch.numel(p) for p in self.parameters() if p.requires_grad)




