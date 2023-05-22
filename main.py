import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import random 
from Preprocess import Prepare_data
import time


class Model(nn.Module) :
    '''
    Our model training class :p
    '''
    def __init__(self, batches, char_idx, embedding_size, hidden_size, num_layers, teacher_forcing_ratio, dropout = 0.2, learning_rate = 0.001, training_ratio = 0.8):
        super().__init__()
        '''
        first we define our models parameters
        '''
        # our optimizer learning rate :
        self.lr = learning_rate

        # our training batches : 
        self.batches = batches

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
        self.opt1 = optim.Adam(self.BILSTM.parameters(), lr = self.lr )
        self.opt2 = optim.Adam([*self.LSTM.parameters(), *self.input_dense.parameters()], lr = self.lr)

        print("all the parameters are initialized")

    

    def word_to_seq(self, word):
        '''
        this function returns a sequence of the unique indexes for the given word 
        (sequence is tensor that can be changed using a .tolist() )
        '''
        word_char_idx_seq =[self.char_index_dic[char] for char in word]    
        return word_char_idx_seq # word sequence
    


    # Let's now build out model :)

    def encode(self, batch):
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
        embedded_word_batch = self.embedding(word_batch)
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
            #tst = torch.squeeze(soft_out, 1)
            #[128, 39]
            #tmp = torch.topk(tst, 3, dim = 1).tolist()
            #topk_indexes.append(tmp)
            outputs.append(soft_out)
        return outputs
    

    def train_model(self, batches, teacher_forcing_bool, epoch):
                
        train_batches = batches           
        epoch_loss = 0
        n = 0            
        test_word = '$' + 'تحليل' + '£'
        for batch in train_batches :
            #print(self.predict(test_word))
            self.opt1.zero_grad()
            self.opt2.zero_grad()
            root_batch, encoder_output, encoder_states = self.encode(batch)
            outputs = self.decode(encoder_states, root_batch, teacher_forcing_bool, epoch)
            a = [torch.squeeze(item, 1) for item in outputs]
            a = [torch.unsqueeze(item, 0) for item in a]
            output = torch.cat(a, dim = 0)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = root_batch.transpose(0, 1)
            trg = trg.reshape(-1)
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
        with torch.no_grad() :
            for batch in val_batches :
                root_batch, encoder_output ,encoder_states = self.encode(batch)
                outputs = self.decode(encoder_output ,encoder_states, root_batch, teacher_forcing_bool, epoch)
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
    
    def predict(self, word):
        '''
        this is the adaptation of encoder-decoder network on a single word w/o optimization
        '''
        
        # Let's turn the word into a sequence of word indexes 
        word_seq = self.word_to_seq(word)
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
        #initialize the input of the decoder
        embedded_char = torch.unsqueeze(self.embedding(torch.tensor(self.char_index_dic[self.sow])), 0)
        prediction_output = [] # a list of the outputs of the decoder 
        # we create a softmax layer : 
        soft = nn.Softmax(dim = 1)
        key_list = list(self.char_index_dic.keys())
        val_list = list(self.char_index_dic.values())
        for i in range(5):              
            decoder_output , (final_hidden, final_cell) = self.LSTM(embedded_char, (final_hidden, final_cell))
            input_dense = nn.Linear(self.hidden_size * 2,self.embedding_size)
            input_decoder_output = input_dense(decoder_output)
            embedded_char = input_decoder_output
            Dense_decoded_output = self.Linear(decoder_output)
            prediction_output.append(soft(Dense_decoded_output).tolist())

        prediction_output = torch.squeeze(torch.tensor(prediction_output), 1)     
        #print(prediction_output.size())
        test_word_seq = word_seq[1:]
        test_word_seq = test_word_seq[:-1]    
        precision = 10     
        top_idx = torch.topk(prediction_output, precision, dim = 1).indices
     
        #if self.char_index_dic[self.sow] in top_idx[0] : 
        init_char = self.char_index_dic[self.sow]
        '''else : 
            init_char = top_idx[0][0]'''
        
        #if self.char_index_dic[self.eow] in top_idx[-1] : 
        final_char = self.char_index_dic[self.eow]
        '''else : 
            final_char = top_idx[-1][0]'''     
        grid = []
        for i in range(precision): 
            for j in range(precision):
                for k in range(precision):
                    tmp = []
                    tmp.append((top_idx[1][i]).item())
                    tmp.append((top_idx[2][j]).item())
                    tmp.append((top_idx[3][k]).item())
                    grid.append(tmp)
        
        # we check the possibilities :      
        best_cases = []
        for case in grid : 
            s = [item for item in case if item in set(test_word_seq)] # we select elts from a that are in l 
            b = [item for item in test_word_seq if item in set(s)] #          
            if s == b and s != [] : 
                best_cases.append(case)
             
        # potential roots :   
        pot_seq = []
        #print(test_word_seq)
        for item in best_cases : 
            #print(item)
            tmp = [init_char] + item  + [final_char]
            pot_seq.append(tmp)           
        final_roots =[]
        for seq in pot_seq : 
            position = [val_list.index(item) for item in seq]
            result_char = [key_list[pos] for pos in position]
            predicted_root = ''.join(result_char)
            final_roots.append(predicted_root)
        return final_roots

    def fit(self, num_epochs):
        """
        let's first prepare our data      
        """
        print(f'The model has {self.count_parameters():,} trainable parameters')
        data = self.batches
        data = random.sample(data, len(data))
        data_size = len(data)
        middle_index = int(data_size * self.ratio)        
        train_batches , val_batches = data[:middle_index], data[middle_index:] 
        epochs = list(range(num_epochs))
        best_val_loss = 1000
        best_model_par = 0
        losses =[]
        predicted_roots = []
        test_word = '$' + 'تحليل' + '£'
  
        for epoch in epochs : 
                
            print('epoch num : ', epoch) 
            t1 = time.time()
            train_batches = random.sample(train_batches , len(train_batches))
            #val_batches = random.sample(val_batches, len(val_batches))    
            train_loss= self.train_model(train_batches, 1, epoch)
            val_loss = self.evaluate_model(val_batches, 0, epoch) # we set the teacher forcing to false            
            t2 = time.time()
            predicted_root = self.predict(test_word)
            print(predicted_root)
            predicted_roots.append(predicted_root)
            tmp = [train_loss, val_loss]
            losses.append(tmp)
            
            print('the training loss : ', train_loss , 'the val loss :', val_loss)
            print('epoch num : ' ,epoch , ' lasted : ', t2 - t1 , 'seconds')
            
            if val_loss < best_val_loss :
                best_val_loss = val_loss 
                best_model_par = self.state_dict()
        torch.save(best_model_par, 'best_model.pt')       
        return losses
    
    def count_parameters(self):
        '''
        function to calculate the total number of parameters in the model
        '''
        return sum(torch.numel(p) for p in self.parameters() if p.requires_grad)




