# -*- encoding: utf-8 -*-
import torch.nn as nn
import numpy as np
import torch
#! /usr/bin/env python
import pickle
import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from input_helpers import InputHelper
from tensorflow.contrib import learn
import gzip
from random import random
# Parameters
# ==================================================


is_char_based=False
word2vec_model="wiki.simple.bin"
word2vec_format="bin"
embedding_dim=300
dropout_keep_prob=1.0
l2_reg_lambda=0.0
hidden_units=50

# Training parameters
batch_size=64
num_epochs=300
evaluate_every=1000
checkpoint_every=1000

# Misc Parameters
allow_soft_placement=True
log_device_placement=False
trainableEmbeddings=False

training_files="train_snli.txt"

max_document_length=15
inpH = InputHelper()
train_set, dev_set, vocab_processor,sum_no_of_batches = inpH.getDataSets(training_files, 10,max_document_length,batch_size, is_char_based)
trainableEmbeddings=False
if is_char_based==True:
    word2vec_model = False

inpH.loadW2V(word2vec_model, word2vec_format)
vocab_size=len(vocab_processor.vocabulary_)
vocab_size



def _calculate_fan_in_and_fan_out(tensor):
    if tensor.ndimension() < 2:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

    if tensor.ndimension() == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = np.prod(tensor.numpy().shape[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the difficulty of training
       deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a normal distribution.
       The resulting tensor will have values sampled from normal distribution with mean=0 and
       std = gain * sqrt(2/(fan_in + fan_out))
    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied
    Examples:
        w = torch.Tensor(3, 5)
        xavier_normal(w, gain=np.sqrt(2.0))
    """

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return tensor.normal_(0, std)


initW = xavier_normal(torch.randn([vocab_size, 300])).numpy()


# initial matrix with random uniform

print("initializing initW with pre-trained word2vec embeddings")


for w in vocab_processor.vocabulary_._mapping:
    arr=[]
    s = re.sub('[^0-9a-zA-Z]+', '', w)
    if w in inpH.pre_emb:
        arr=inpH.pre_emb[w]
    elif w.lower() in inpH.pre_emb:
        arr=inpH.pre_emb[w.lower()]
    elif s in inpH.pre_emb:
        arr=inpH.pre_emb[s]
    elif s.isdigit():
        arr=inpH.pre_emb["zero"]
    if len(arr)>0:
        idx = vocab_processor.vocabulary_.get(w)
        initW[idx]=np.asarray(arr).astype(np.float32)
def sample(batches):
    batch=batches.__next__()
    x1_batch,x2_batch, y_batch = zip(*batch)
    x1_batch=torch.tensor(x1_batch)
    x2_batch=torch.tensor(x2_batch)
    y_batch=np.array(y_batch)
    y_batch=torch.tensor(y_batch).view(-1)
    return x1_batch,x2_batch,y_batch



class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size,embedding_dims,hidden_dims,num_layers,batch_size,dropout_p,is_train):
        super(LSTMEncoder, self).__init__()
        self.vocab_size=vocab_size
        self.hidden_dims=hidden_dims
        self.embedding_dims=embedding_dims
        self.num_layers=num_layers
        self.dropout_p = dropout_p
        self.is_train=is_train
        self.batch_size=batch_size
        
        
        self.embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dims,padding_idx=0, max_norm=None, 
                                       scale_grad_by_freq=False, sparse=False)
       # print(self.embedding_table.weight)
        self.embedding_table.weight.requires_grad=self.is_train
        #print(self.embedding_table.weight.requires_grad)
        self.lstm_rnn = nn.LSTM(input_size=self.embedding_dims,hidden_size=self.hidden_dims, num_layers=1)
     #   print(lstm_rnn..requires_grad)
        self.dropout = nn.Dropout(self.dropout_p )
        
    def initHidden(self):
        hidden_a = torch.randn(1, self.batch_size,self.hidden_dims)
        hidden_b = torch.randn(1,self.batch_size,self.hidden_dims)
        return (hidden_a,hidden_b)
        
    def forward(self,data,batch_size,hidden):
        output = self.embedding_table(data).view(-1,batch_size,embedding_dims)

        output, hidden= self.lstm_rnn(output, hidden)
        return output, hidden
    
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
class SiameseClassifier(nn.Module):
    def __init__(self, vocab_size,embedding_dims,hidden_dims,num_layers,batch_size,dropout_p,pretrained_embeddings,is_train,learning_rate):
        super(SiameseClassifier, self).__init__()
        self.vocab_size=vocab_size
        self.hidden_dims=hidden_dims
        self.embedding_dims=embedding_dims
        self.num_layers=num_layers
        self.dropout_p = dropout_p
        self.is_train=is_train
        self.learning_rate=learning_rate
        self.batch_size=batch_size
 
        
        self.encoder_a = self.encoder_b = LSTMEncoder(self.vocab_size,self.embedding_dims,self.hidden_dims,self.num_layers,self.batch_size,self.dropout_p,self.is_train)
        # Initialize pre-trained embeddings, if given
        self.initialize_parameters()
        
        if not self.is_train:
              self.encoder_a.embedding_table.weight.data.copy_(pretrained_embeddings)
        
        self.pretrained_embeddings=self.encoder_a.embedding_table.weight

        # Initialize network parameters
      

        # Initialize network optimizers
        self.optimizer_a = optim.Adam(filter(lambda p: p.requires_grad, self.encoder_a.parameters()), lr=self.learning_rate,
                                      betas=(0.25, 0.999))
        self.optimizer_b = optim.Adam(filter(lambda p: p.requires_grad,self.encoder_b.parameters()), lr=self.learning_rate,betas=(0.25, 0.999))
        
        print("Model Compile")
    
    def forward(self,x1,x2,y):
        """ Performs a single forward pass through the siamese architecture. """
        # Checkpoint the encoder state
        self.x1=x1
        self.x2=x2
        self.y=y
        state_dict = self.encoder_a.state_dict()

        # Obtain sentence encodings from each encoder
        hidden_a= self.encoder_a.initHidden()
        
        output_a,hidden_a=self.encoder_b(self.x1,self.batch_size,hidden_a)
            

        # Restore checkpoint to establish weight-sharing
        self.encoder_b.load_state_dict(state_dict)
        hidden_b= self.encoder_b.initHidden()
        
        output_b,hidden_b=self.encoder_b(self.x2,self.batch_size,hidden_b)
        
        encoding_a=hidden_a[0]
        encoding_b=hidden_b[0] 
       

        # Format sentence encodings as 2D tensors
        self.encoding_a = encoding_a.squeeze(dim=0)  #[batch, hidden]
        self.encoding_b = encoding_b.squeeze(dim=0)
 

    def get_loss(self):
        y=self.y.float()
        dist_sq =torch.sqrt(torch.sum(torch.pow(self.encoding_a-self.encoding_b , 2), 1))
        dist_1=torch.sqrt(torch.sum(torch.pow(self.encoding_a,2),1))
        dist_2=torch.sqrt(torch.sum(torch.pow(self.encoding_b,2),1))
        dist=torch.div(dist_sq,dist_1+dist_2)
        self.dist=dist
        loss=y*(torch.pow(dist,2))+(1-y)*torch.clamp(1-dist, min=0.0)
        loss=torch.sum(loss)/2.0/dist.size()[0]
        
        self.loss=loss
    def get_accuracy(self):
        a=np.rint(self.dist.data.numpy())
        b=np.ones(self.batch_size)
        c=np.equal(b-a, self.y.data.numpy())
        g=[1 if x else 0 for x in c ]
        accu=np.sum(g)/batch_size
        self.accuracy=accu

    def initialize_parameters(self):
        """ Initializes network parameters. """
        state_dict = self.encoder_a.state_dict()
        for key in state_dict.keys():
            if '.weight' in key:
                state_dict[key] = xavier_normal(state_dict[key])
            if '.bias' in key:
                bias_length = state_dict[key].size()[0]
                start, end = bias_length // 4, bias_length // 2
                state_dict[key][start:end].fill_(2.5)
        self.encoder_a.load_state_dict(state_dict)
    def train_step(self, train_batch_a, train_batch_b, train_labels):

        # Get gradients
        self.encoder_a.zero_grad()  
        self.forward(train_batch_a, train_batch_b, train_labels)
       # encoder_a == encoder_b
        self.get_loss()
        self.get_accuracy()
        self.loss.backward()

        # Clip gradients
        clip_grad_norm(filter(lambda p: p.requires_grad,self.encoder_b.parameters()), 0.25)
        
        # Optimize
        self.optimizer_a.step()
    def dev_step(self, train_batch_a, train_batch_b, train_labels):

        # Get gradients
        self.encoder_a.zero_grad()  
        self.forward(train_batch_a, train_batch_b, train_labels)
       # encoder_a == encoder_b
        self.get_loss()
        self.get_accuracy()
        #self.loss.backward()

        # Clip gradients
        #clip_grad_norm(filter(lambda p: p.requires_grad,self.encoder_b.parameters()), 0.25)
        
        # Optimize
       # self.optimizer_a.step()


from utils.init_and_storage import add_pretrained_embeddings, extend_embeddings, update_learning_rate, save_network

import os

save_dir = '/models'
pretraining_dir = '/models'

start_early_stopping=90
start_annealing=3
save_freq=5
best_validation_accuracy=0
patience=3
epochs_without_improvement=0

batch_size=32
num_epochs=100

vocab_size=31337
hidden_dims=50
embedding_dims=300
num_layers=3
dropout_p = 1.
learning_rate=0.0001
old_learning_rate=0.0001

is_train=False

batch_size=32
pretrained_embeddings=torch.tensor(initW)
report_freq=200
final_epoch=0 
model=SiameseClassifier(vocab_size,embedding_dims,hidden_dims,num_layers,batch_size,dropout_p,pretrained_embeddings,is_train,learning_rate)

#subset=10000
subset=train_set[0].shape[0]+1
for epoch in range(num_epochs):
    batches=inpH.batch_iter(list(zip(train_set[0][0:subset], train_set[1][0:subset], train_set[2][0:subset])), batch_size, 1)
    running_loss = list()
    total_train_loss = list()
    running_accu=list()
    total_accu=list()
    for i in range(int(train_set[0][0:subset].shape[0]/batch_size)):
        
        x1,x2,y=sample(batches)
        model.train_step(x1,x2,y)
        train_batch_loss =model.loss.data[0]
    
        running_loss.append(train_batch_loss)
        total_train_loss.append(train_batch_loss)
        batch_accu=model.accuracy
        running_accu.append(batch_accu)
        total_accu.append(batch_accu)
        if((i+1)%report_freq==0):
            running_avg_loss = sum(running_loss) / len(running_loss)
            running_avg_accu=sum(running_accu)/len(running_accu)
            print('Epoch: %d | Training Batch: %d | Average loss since batch %d: %.4f | Model accuracy: %.4f ' %
                  (epoch+1, i+1, i - report_freq+1, running_avg_loss,running_avg_accu))
            running_loss = list()
            running_accu=list()
    avg_training_loss = sum(total_train_loss) / len(total_train_loss)
    avg_training_accuracy = sum(total_accu) / len(total_accu)
    print('Average training batch loss at epoch %d: %.4f  batch accuracy:  %.4f' % (epoch, avg_training_loss, avg_training_accuracy))

     # Validate after each epoch; set tracking variables
    if epoch >= start_early_stopping:
        total_valid_loss = list()
        # Initiate the training data loader
        batches=inpH.batch_iter(list(zip(dev_set[0], dev_set[1], dev_set[2])), batch_size, 1)

        # Validation loop (i.e. perform inference on the validation set)
        for i in range(int(dev_set[0].shape[0]/batch_size)):
            x1,x2,y=sample(batches)
            # Get predictions and update tracking values
            model.dev_step(x1, x2, y)
            valid_batch_loss = model.loss.data[0]
            total_valid_loss.append(valid_batch_loss)

        # Report fold statistics
        avg_valid_accuracy = sum(total_valid_loss) / len(total_valid_loss)
        print('Average validation fold accuracy at epoch %d: %.4f' % (epoch, avg_valid_accuracy))
        # Save network parameters if performance has improved
        if avg_valid_accuracy <= best_validation_accuracy:
            epochs_without_improvement += 1
        else:
            best_validation_accuracy = avg_valid_accuracy
            epochs_without_improvement = 0
            
    # Save network parameters at the end of each nth epoch
    if epoch % save_freq == 0 and epoch != 0:
        print('Saving model networks after completing epoch %d' % epoch)
        save_network(model.encoder_a, 'sim_classifier', epoch, save_dir)

    # Anneal learning rate:
    if epochs_without_improvement == start_annealing:
        old_learning_rate = model.learning_rate
        learning_rate *= annealing_factor
        update_learning_rate(model.optimizer_a, learning_rate)
        print('Learning rate has been updated from %.4f to %.4f' % (old_learning_rate, learning_rate))

    # Terminate training early, if no improvement has been observed for n epochs
    if epochs_without_improvement >= patience:
        print('Stopping training early after %d epochs, following %d epochs without performance improvement.' %
              (epoch, epochs_without_improvement))
        final_epoch = epoch
        save_network(model.encoder_a, 'sim_classifier', 'latest', save_dir)
        break



print('Training procedure concluded after %d epochs total. Best validated epoch: %d.'
      % (final_epoch, final_epoch - patience))
if model.is_train:
    # Save pretrained embeddings and the vocab object
    pretrained_path = os.path.join(save_dir, 'pretrained.pkl')
    pretrained_embeddings = model.pretrained_embeddings.data
    with open(pretrained_path, 'wb') as f:
        pickle.dump((pretrained_embeddings), f)
    print('Pre-trained parameters saved to %s' % pretrained_path)
    
print("Done")



