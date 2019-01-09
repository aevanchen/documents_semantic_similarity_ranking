from __future__ import print_function
import os
import sys
import time
import json
import argparse
import random
random.seed(49999)
import numpy as np
random.seed(49999)
import tensorflow
tensorflow.set_random_seed(49999)
from collections import OrderedDict
import keras
import keras.backend as K
from keras.models import Sequential, Model
sys.path.append('/home/xingyuchen/jupyter/matchzoo/utils/')
sys.path.append('/home/xingyuchen/jupyter/matchzoo/inputs/')
from preprocess import *
from rank_io import *
from utils import *
import inputs
import metrics
from losses import *
from optimizers import *
from tqdm import tqdm
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config = config)


#load model
model_file =  "/home/xingyuchen/jupyter/matchzoo/anmm_ranking.config"
with open(model_file, 'r') as f:
    config = json.load(f)
global_conf = config["global"]
optimizer = global_conf['optimizer']
optimizer=optimizers.get(optimizer)
K.set_value(optimizer.lr, global_conf['learning_rate'])
weights_file = str(global_conf['weights_file']) + '.%d'
display_interval = int(global_conf['display_interval'])
num_iters = int(global_conf['num_iters'])
save_weights_iters = int(global_conf['save_weights_iters'])
test_weights_iters=int(global_conf['test_weights_iters'])
# read input config
input_conf = config['inputs']
share_input_conf = input_conf['share']




# collect embedding

embed_dict = read_embedding(filename=share_input_conf['embed_path'])
_PAD_ = share_input_conf['vocab_size'] - 1
embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
embed = np.float32(np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
print('[Embedding] Embedding Load Done.', end='\n')

# list all input tags and construct tags config
input_train_conf = OrderedDict()
input_eval_conf = OrderedDict()
for tag in input_conf.keys():
    if 'phase' not in input_conf[tag]:
        continue
    if input_conf[tag]['phase'] == 'TRAIN':
        input_train_conf[tag] = {}
        input_train_conf[tag].update(share_input_conf)
        input_train_conf[tag].update(input_conf[tag])
    elif input_conf[tag]['phase'] == 'EVAL':
        input_eval_conf[tag] = {}
        input_eval_conf[tag].update(share_input_conf)
        input_eval_conf[tag].update(input_conf[tag])
print('[Input] Process Input Tags. %s in TRAIN, %s in EVAL.' % (input_train_conf.keys(), input_eval_conf.keys()), end='\n')

# collect dataset identification
dataset = {}
for tag in input_conf:
    if tag != 'share' and input_conf[tag]['phase'] == 'PREDICT':
        continue
    if 'text1_corpus' in input_conf[tag]:
        datapath = input_conf[tag]['text1_corpus']
        if datapath not in dataset:
            dataset[datapath], _ = read_data(datapath)
    if 'text2_corpus' in input_conf[tag]:
        datapath = input_conf[tag]['text2_corpus']
        if datapath not in dataset:
            dataset[datapath], _ = read_data(datapath)
print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')






# initial data generator
train_gen = OrderedDict()
eval_gen = OrderedDict()

for tag, conf in input_train_conf.items():
    print(conf, end='\n')
    conf['data1'] = dataset[conf['text1_corpus']]
    conf['data2'] = dataset[conf['text2_corpus']]
    generator = inputs.get(conf['input_type'])
    train_gen[tag] = generator( config = conf )

for tag, conf in input_eval_conf.items():
    print(conf, end='\n')
    conf['data1'] = dataset[conf['text1_corpus']]
    conf['data2'] = dataset[conf['text2_corpus']]
    generator = inputs.get(conf['input_type'])
    eval_gen[tag] = generator( config = conf )
    
import models
sys.path.append('/home/xingyuchen/jupyter/matchzoo/models/')
from matchpyramid import *
from anmm import *
model_config = config['model']['setting']
model_config.update(config['inputs']['share'])
sys.path.insert(0, config['model']['model_path'])

model= ANMM( model_config).build()



loss = []
for lobj in config['losses']:
    if lobj['object_name'] in mz_specialized_losses:
        loss.append(rank_losses.get(lobj['object_name'])(lobj['object_params']))
    else:
        loss.append(rank_losses.get(lobj['object_name']))
eval_metrics = OrderedDict()
for mobj in config['metrics']:
    mobj = mobj.lower()
    if '@' in mobj:
        mt_key, mt_val = mobj.split('@', 1)
        eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
    else:
        eval_metrics[mobj] = metrics.get(mobj)
model.compile(optimizer=optimizer, loss=loss)
print('[Model] Model Compile Done.', end='\n')


def evaluate():
      i=0
      for tag, generator in eval_gen.items():
        if(i==1):
            continue
        genfun = generator.get_batch_generator()
        print('[%s]\t[Eval:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
        res = dict([[k,0.] for k in eval_metrics.keys()])
        num_valid = 0
        i=0
        for input_data, y_true in genfun:
            
            y_pred = model.predict(input_data, batch_size=len(y_true))
            if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                list_counts = input_data['list_counts']
                for k, eval_func in eval_metrics.items():
                    for lc_idx in range(len(list_counts)-1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx+1]
                        res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])
                num_valid += len(list_counts) - 1
            else:
                for k, eval_func in eval_metrics.items():
                    res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                num_valid += 1
        i=i+1
        generator.reset()
        print('Iter:%d\t%s' % (i_e, '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()])), end='\n')
        sys.stdout.flush()
num_iters=100
save_weights_iters=10
test_weights_iters=10
display_interval=20000

for i_e in range(num_iters):
    for tag, generator in train_gen.items():
        genfun = generator.get_batch_generator()
        print(type(generator))
        print('[%s]\t[Train:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
        history = model.fit_generator(
                genfun,
                steps_per_epoch = display_interval,
                #steps_per_epoch=1000,
                epochs = 1,
                shuffle=False,
                verbose = 1
            ) #callbacks=[eval_map])
        print('Iter:%d\tloss=%.6f' % (i_e, history.history['loss'][0]), end='\n')
    if((i_e+1)%test_weights_iters==0):
        evaluate()
        
  
    if (i_e+1) % save_weights_iters == 0:
        model.save_weights(weights_file % (i_e+1))
print("Done")