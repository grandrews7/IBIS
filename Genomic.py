#!/usr/bin/env python
# coding: utf-8

# In[89]:


#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
tf.__version__
tf.compat.v1.disable_eager_execution()

import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras.layers import Input, Lambda, Conv1D, maximum, GlobalMaxPooling1D, Dense, GaussianNoise, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import pandas as pd
import logomaker

import random
from tqdm import trange
from subprocess import Popen, PIPE, run
import sys
import pickle
from pyfaidx import Fasta
from  tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence
import random
import glob
import bioframe
import os


# In[90]:


print(tf.__version__)


# In[91]:


#Models
def construct_model(num_kernels=32,
                    kernel_width=24,
                    seq_len=None,
                    dropout_prop=0.0,
                    use_bias=False,
                    kernel_initializer=initializers.RandomNormal(stddev=0.0001, seed=12),
                    optimizer='adam',
                    activation='linear',
                    num_classes=1,
                    l1_reg=0.0,
                    l2_reg= 0.0,
                    gaussian_noise = 0.1,
                    spatial_dropout = 0.0,
                    rc = True,
                    padding="same",
                    conv_name="shared_conv"):
    if rc:
        seq_input = Input(shape=(seq_len,4))
        rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
        seq_rc = rc_op(seq_input)
        if gaussian_noise > 0.0:
            noisy_seq = GaussianNoise(gaussian_noise)(seq_input)
            noisy_seq_rc = rc_op(noisy_seq)
        
        shared_conv = Conv1D(num_kernels, kernel_width,
                             strides=1, padding=padding, 
                             activation=activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=regularizers.l1_l2(l1=l1_reg,
                                                                   l2=l2_reg),
                             bias_initializer='zeros',
                             name=conv_name)

        if gaussian_noise > 0:
            conv_for = shared_conv(noisy_seq)
            conv_rc = shared_conv(noisy_seq_rc)
        else:
            conv_for = shared_conv(seq_input)
            conv_rc = shared_conv(seq_rc)
            

        merged = maximum([conv_for, conv_rc])
        pooled = GlobalMaxPooling1D()(merged)
        if dropout_prop > 0.0:
            dropout = Dropout(dropout_prop)(pooled)
            output = Dense(1, activation='sigmoid',
                       use_bias=True,
                       kernel_initializer=initializers.RandomUniform(minval=0.0, maxval=0.001, seed=12), 
                       kernel_constraint=non_neg(), 
                       bias_initializer='zeros',
                       name="dense_1")(dropout)
        else:
            output = Dense(1, activation='sigmoid',
                           use_bias=True,
                           kernel_initializer=initializers.RandomUniform(minval=0.0, maxval=0.001, seed=12), 
                           kernel_constraint=non_neg(), 
                           bias_initializer='zeros',
                           name="dense_1")(pooled)
        model = Model(inputs=seq_input, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model



def construct_scan_model(conv_weights):
    kernel_width = conv_weights.shape[0]
    num_kernels = conv_weights.shape[2]
    seq = Input(shape=(None,4))
    conv = Conv1D(num_kernels, kernel_width, 
                  name = 'scan_conv',
                  strides=1, 
                  padding='valid', 
                  activation='linear', 
                  use_bias=False, 
                  kernel_initializer='zeros', 
                  bias_initializer='zeros',
                  trainable=False)
    
    conv_seq = conv(seq)
    
    
    model = Model(inputs=seq, outputs=conv_seq)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.get_layer('scan_conv').set_weights([conv_weights])
    return model


def construct_score_model(conv_weights):
    kernel_width = conv_weights.shape[0]
    num_kernels = conv_weights.shape[2]
    seq = Input(shape=(None,4))
    rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    seq_rc = rc_op(seq)
    
    conv = Conv1D(num_kernels, kernel_width, 
                  name = 'score_conv',
                  strides=1, 
                  padding='valid', 
                  activation='linear', 
                  use_bias=use_bias, 
                  kernel_initializer='zeros', 
                  bias_initializer='zeros',
                  trainable=False)
    
    conv_for = conv(seq)
    conv_rc = conv(seq_rc)
    
    merged = maximum([conv_for, conv_rc])
    pooled = GlobalMaxPooling1D()(merged)
    model = Model(inputs=seq, outputs=pooled)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.get_layer("score_conv").set_weights([conv_weights])
    print(model.summary())
    return model


# In[10]:


# altschulEriksonDinuclShuffle.py
# P. Clote, Oct 2003

def computeCountAndLists(s):

    #Initialize lists and mono- and dinucleotide dictionaries
    List = {} #List is a dictionary of lists
    List['A'] = []; List['C'] = [];
    List['G'] = []; List['T'] = [];
    # FIXME: is this ok?
    List['N'] = []
    nuclList   = ["A","C","G","T","N"]
    s       = s.upper()
    #s       = s.replace("U","T")
    nuclCnt    = {}  #empty dictionary
    dinuclCnt  = {}  #empty dictionary
    for x in nuclList:
        nuclCnt[x]=0
        dinuclCnt[x]={}
        for y in nuclList:
            dinuclCnt[x][y]=0

    #Compute count and lists
    nuclCnt[s[0]] = 1
    nuclTotal     = 1
    dinuclTotal   = 0
    for i in range(len(s)-1):
        x = s[i]; y = s[i+1]
        List[x].append( y )
        nuclCnt[y] += 1; nuclTotal  += 1
        dinuclCnt[x][y] += 1; dinuclTotal += 1
    assert (nuclTotal==len(s))
    assert (dinuclTotal==len(s)-1)
    return nuclCnt,dinuclCnt,List


def chooseEdge(x,dinuclCnt):
    z = random.random()
    denom=dinuclCnt[x]['A']+dinuclCnt[x]['C']+dinuclCnt[x]['G']+dinuclCnt[x]['T']+dinuclCnt[x]['N']
    numerator = dinuclCnt[x]['A']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['A'] -= 1
        return 'A'
    numerator += dinuclCnt[x]['C']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['C'] -= 1
        return 'C'
    numerator += dinuclCnt[x]['G']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['G'] -= 1
        return 'G'
    numerator += dinuclCnt[x]['T']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['T'] -= 1
        return 'T'
    dinuclCnt[x]['N'] -= 1
    return 'N'

def connectedToLast(edgeList,nuclList,lastCh):
    D = {}
    for x in nuclList: D[x]=0
    for edge in edgeList:
        a = edge[0]; b = edge[1]
        if b==lastCh: D[a]=1
    for i in range(3):
        for edge in edgeList:
            a = edge[0]; b = edge[1]
            if D[b]==1: D[a]=1
    ok = 0
    for x in nuclList:
        if x!=lastCh and D[x]==0: return 0
    return 1

def eulerian(s):
    nuclCnt,dinuclCnt,List = computeCountAndLists(s)
    #compute nucleotides appearing in s
    nuclList = []
    for x in ["A","C","G","T","N"]:
        if x in s: nuclList.append(x)
    #create dinucleotide shuffle L
    firstCh = s[0]  #start with first letter of s
    lastCh  = s[-1]
    edgeList = []
    for x in nuclList:
        if x!= lastCh: edgeList.append( [x,chooseEdge(x,dinuclCnt)] )
    ok = connectedToLast(edgeList,nuclList,lastCh)
    return ok,edgeList,nuclList,lastCh


def shuffleEdgeList(L):
    n = len(L); barrier = n
    for i in range(n-1):
        z = int(random.random() * barrier)
        tmp = L[z]
        L[z]= L[barrier-1]
        L[barrier-1] = tmp
        barrier -= 1
    return L

def dinuclShuffle(s):
    ok = 0
    while not ok:
        ok,edgeList,nuclList,lastCh = eulerian(s)
    nuclCnt,dinuclCnt,List = computeCountAndLists(s)

    #remove last edges from each vertex list, shuffle, then add back
    #the removed edges at end of vertex lists.
    for [x,y] in edgeList: List[x].remove(y)
    for x in nuclList: shuffleEdgeList(List[x])
    for [x,y] in edgeList: List[x].append(y)

    #construct the eulerian path
    L = [s[0]]; prevCh = s[0]
    for i in range(len(s)-2):
        ch = List[prevCh][0]
        L.append( ch )
        del List[prevCh][0]
        prevCh = ch
    L.append(s[-1])
    #t = string.join(L,"")
    t = "".join(L)
    return t


def get_information_content(x):
    ic = x * np.log2((x + .001) / .25)
    if ic > 0:
        return(ic)
    else:
        return(0.0)
    
def get_info_content(ppm):
    w = ppm.shape[0]
    info = np.zeros(w)
    for i in range(w):
        for j in range(4):
            info[i] += ppm[i,j] * np.log2((ppm[i,j] + .001) / 0.25)
    return(info)
    
def trim_ppm(ppm, min_info=0.0):
    info = get_info_content(ppm)
    start_index = 0
    w = ppm.shape[0]
    stop_index = w
    for i in range(w):
        if info[i] < min_info:
            start_index += 1
        else:
            break

    for i in range(w):
        if info[w-i-1] < 0.25:
            stop_index -= 1
        else:
            break

    if np.max(info) < 0.25:
        return(ppm, 0, w)
    else:
        return(ppm[start_index:stop_index,:], start_index, stop_index)
    
DNA_SEQ_DICT = {
    'A' : [1, 0, 0, 0],
    'C' : [0, 1, 0, 0],
    'G' : [0, 0, 1, 0],
    'T' : [0, 0, 0, 1],
}

def encode_sequence(seq, N = [0, 0, 0, 0], seq_dict = None, useN = None):
    if seq_dict is None:
        seq_dict = DNA_SEQ_DICT
    if useN == 'uniform':
        N = [(1/len(seq_dict)) for _ in seq_dict]
    elif useN == 'zeros':
        N = [0 for _ in seq_dict]
    d = { **seq_dict, 'N' : N }
    return np.array([d[nuc] for nuc in list(seq)]).astype('float32')
 
def decode_sequence(encoded_seq, seq_dict = None):
    if seq_dict is None:
        seq_dict = DNA_SEQ_DICT
    seq_list = encoded_seq.astype('int').tolist()
    def decode_base(encoded_base):
        for letter,onehot in seq_dict.items():
            if np.array_equal(encoded_base, onehot):
                return letter
        return "N"
    return "".join(decode_base(b) for b in encoded_seq.astype('int'))


# In[11]:


from  tensorflow.keras.callbacks import Callback
class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.

    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2,
                 shape="cosine"):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor
        
        self.shape = shape
        self.history = {}
        self.learning_rates = []

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        #print(fraction_to_restart)
        if self.shape == "cosine":
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        else:
            if fraction_to_restart < 0.5:
                lr = fraction_to_restart * (self.max_lr - self.min_lr) / 0.5 + self.min_lr
            else:
                lr = (1 - fraction_to_restart) * (self.max_lr - self.min_lr) / 0.5 + self.min_lr
        self.learning_rates.append(lr)
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        if self.shape == "cosine":
            K.set_value(self.model.optimizer.lr, self.max_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)
        
class SWA(Callback):

    def __init__(self, epochs_to_train, prop = 0.2, interval = 1):
        super(SWA, self).__init__()
        self.epochs_to_train = epochs_to_train
        self.prop = prop
        self.interval = interval
        self.n_models = 0
        self.epoch = 0
        
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        self.weights = []
    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        if epoch % self.interval == 0:
            self.weights.append(self.model.get_weights())
            self.n_models += 1
        else:
            pass

    def on_train_end(self, logs=None):
        num_models_to_average = int(np.ceil(self.prop * self.epoch))
        new_weights = list()
        for weights_list_tuple in zip(*self.weights[-num_models_to_average:]): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        self.model.set_weights(new_weights)


# In[154]:


class dataGen(Sequence):
    def __init__(self, posSeqs, 
                 posWeights = None,
                 negSeqs=None,
                 batchSize = 32,
                 seqsPerEpoch=10000,
                 padBy = 24):
        
        self.posSeqs = posSeqs
        self.posWeights = posWeights
        self.nPos = len(self.posSeqs)
        self.seqsPerEpoch = seqsPerEpoch
        if self.nPos < (self.seqsPerEpoch / 2):
            if self.posWeights is None:
                self.posSeqs = list(np.random.choice(self.posSeqs, int(self.seqsPerEpoch / 2)))
                self.nPos = len(self.posSeqs)
            else:
                tmp = list(zip(self.posSeqs, self.posWeights))
                tmp = random.choices(tmp, k=int(self.seqsPerEpoch / 2))
                self.posSeqs, self.posWeights = map(list, zip(*tmp))
            
        if negSeqs is not None:
            self.negSeqs = negSeqs
            self.nNeg = len(self.negSeqs)
        else:
            self.negSeqs = None
            
        
        if self.negSeqs is not None:
            self.L = np.max([len(x) for x in self.posSeqs + self.negSeqs])
        else:
            self.L = np.max([len(x) for x in self.posSeqs])
        
        self.batchSize = batchSize
        self.b2 = self.batchSize // 2
        self.padBy = padBy
        
        self.labels = np.array([1 for i in range(self.b2)] + [0 for i in range(self.b2)])
        self.epoch = 0
        self.nIter = 0
        self.shuffleEvery = int(self.nPos // (self.seqsPerEpoch / 2) )      
        print(self.shuffleEvery)
    
    def __len__(self):
        return(int(np.floor(self.seqsPerEpoch / self.batchSize)))
    
    def on_train_begin(self):
        tmp = list(zip(self.posSeqs, self.posWeights))
        tmp = sorted(tmp, key=lambda x: x[1], ascending=False)
        print(tmp[:10])
        self.posSeqs, self.posWeights = map(list, zip(*tmp))
        
    def on_epoch_end(self):
        if self.epoch == self.shuffleEvery:
            print("Shuffling positive sequences")
            if self.posWeights is None:
                random.shuffle(self.posSeqs)
            else:
                tmp = list(zip(self.posSeqs, self.posWeights))
                random.shuffle(tmp)
                self.posSeqs, self.posWeights = map(list, zip(*tmp))

    
    def __getitem__(self, index):
        idx = (self.nIter * self.b2) % (self.nPos - self.b2)
        posSample = self.posSeqs[idx:idx+self.b2]
        if self.posWeights is not None:
            posWeights = self.posWeights[idx:idx+self.b2]
            
        self.nIter += 1
        
        if self.negSeqs is not None:
            negSample = random.sample(self.negSeqs, self.b2)
        else:
            negSample = [dinuclShuffle(x) for x in posSample]
            if self.posWeights is not None:
                # negWeights = posWeights
                # negWeights = [1.0 for i in range(self.b2)]
                negWeights = posWeights
                
            
        X = 0.25 * np.ones((self.batchSize, 2*self.padBy + self.L, 4))
        if self.posWeights is not None:
            weights = posWeights + negWeights
            
        for i,seq in enumerate(posSample + negSample):
            l = len(seq)
            start = self.padBy + (self.L - l) // 2
            stop = start + l
            X[i,start:stop,:] = encode_sequence(seq)
            
        if self.posWeights is not None:
            return(X, self.labels, np.array(weights))
        else:
            return(X, self.labels)


# In[182]:


# In[92]:


def get_rc(re):
    """
    Return the reverse complement of a DNA/RNA RE.
    """
    return re.translate(str.maketrans('ACGTURYKMBVDHSWN', 'TGCAAYRMKVBHDSWN'))[::-1]


def count_seqs_with_words(seqs, halflength, ming, maxg, alpha, revcomp, desc):
    if alpha == 'protein':
        ambiguous_character = 'X'
    else:
        ambiguous_character = 'N'
    gapped_kmer_dict = {}  # each key is the gapped k-mer word
    for g in trange(ming, maxg + 1, 1, desc=desc):
        w = g+2*halflength # length of the word
        gap = g * ambiguous_character
        for seq in seqs:
            slen = len(seq)
            for i in range(0, slen-w+1):
                word = seq[i : i+w]
                # skip word if it contains an ambiguous character
                if ambiguous_character in word:
                    continue
                # convert word to a gapped word. Only the first and last half-length letters are preserved
                word = word[0:halflength] + gap + word[-halflength:]
                update_gapped_kmer_dict(gapped_kmer_dict, word, revcomp)
    return gapped_kmer_dict


def update_gapped_kmer_dict(gapped_kmer_dict, word, revcomp):
    # use the lower alphabet word for rc
    if revcomp:
        word = min(word, get_rc(word))
    if word in gapped_kmer_dict:  # word has been encountered before, add 1
        gapped_kmer_dict[word] += 1
    else:  # word has not been encountered before, create new key
        gapped_kmer_dict[word] = 1


def get_zscores(pos_seq_counts, neg_seq_counts):
    zscores_dict = {}
    for word in pos_seq_counts:
        p = pos_seq_counts[word]
        if word in neg_seq_counts:
            n = neg_seq_counts[word]
        else:
            n = 1
        zscore = 1.0*(p - n)/np.sqrt(n)
        zscores_dict[word] = zscore
    return zscores_dict


# returns the words in order, from largest to smallest, by z-scores
def sorted_zscore_keys(zscores_dict):
    sorted_keys = sorted(zscores_dict, key=zscores_dict.__getitem__, reverse=True)
    return sorted_keys


def find_n_top_words(zscores_dict, num_find):
    keys = np.array(list(zscores_dict.keys()))
    values = np.array(list(zscores_dict.values()))
    ind = np.argpartition(values, -num_find)[-num_find:]
    top_words = list(keys[ind])
    return top_words


def find_enriched_gapped_kmers(pos_seqs, neg_seqs, halflength, ming, maxg, alpha, revcomp, num_find):
    pos_seq_counts = count_seqs_with_words(pos_seqs, halflength, ming, maxg, alpha, revcomp,
                                           'Searching positive sequences')
    neg_seq_counts = count_seqs_with_words(neg_seqs, halflength, ming, maxg, alpha, revcomp,
                                           'Searching negative sequences')
    zscores = get_zscores(pos_seq_counts,neg_seq_counts)
    top_words = find_n_top_words(zscores, num_find)
    return top_words


def load_peaks(peaks):
    data = pd.read_csv(peaks, header=0, sep="\t", skipfooter=1)
    data = data.rename(columns = {"#CHROM" : "chrom", 
                                  "START" : "start", 
                                  "END" : "end", 
                                  "fold_enrichment" : "signal",
                                  " supporting_peakcallers" : "peak_callers"})
    data["n_peak_callers"] = data.peak_callers.str.count(",") + 1
    return(data)

def merge_peaks(peaks_dfs):
    toReturn = peaks_dfs[0].copy()
    for i in range(1, len(peaks_dfs)):
        toReturn = bioframe.closest(toReturn, peaks_dfs[i], 
                               suffixes=('_1','_2'))
        toReturn = toReturn[toReturn["distance"] == 0]
        toReturn["abs_summit_1"] = (i * toReturn["abs_summit_1"] + toReturn["abs_summit_2"]) // (i+1)
        toReturn["signal_1"] = (i * toReturn["signal_1"] + toReturn["signal_2"]) / (i+1)
        toReturn = toReturn.rename(columns = {"chrom_1" : "chrom", 
                                              "start_1" : "start", 
                                              "end_1" : "end", 
                                              "signal_1" : "signal",
                                              "abs_summit_1" : "abs_summit"})
    
    return(toReturn)


# TF = "GABPA"
# assay = "CHS"

TF = sys.argv[1]
assay = sys.argv[2]

# In[204]:


w = 32
num_kernels = 16
use_bias = False
epochs=1000
w2 = 50

prefix = TF + "-" + assay
if assay == "All":
    dataFiles = glob.glob("/home/gregory.andrews-umw/IBIS/data/*/" + TF + "/*" )
    data = [load_peaks(_) for _ in dataFiles]
    data = pd.concat(data)
    data = data.rename(columns = {"fold_enrichment" : "signal"})
    data = data[["chrom", "start", "end", "abs_summit", "signal"]]

elif assay == "Both":
    CHS = glob.glob("/home/gregory.andrews-umw/IBIS/data/" + "CHS" + "/" + TF + "/*" )
    GHTS = glob.glob("/home/gregory.andrews-umw/IBIS/data/" + "GHTS" + "/" + TF + "/*" )
    cycles = sorted([os.path.basename(_).split(".")[-2] for _ in GHTS])
    max_cycle = cycles[-1]
    GHTS = [_ for _ in GHTS if "." + max_cycle + "." in _]
    GHTS_data = [load_peaks(_) for _ in GHTS]
    GHTS_data = pd.concat(GHTS_data)
    GHTS_data = GHTS_data[["chrom", "start", "end", "abs_summit", "signal"]]
    
    CHS_data = [load_peaks(_) for _ in CHS]
    CHS_data = merge_peaks(CHS_data)
    CHS_data = CHS_data[["chrom", "start", "end", "abs_summit", "signal"]]
    data = pd.concat([CHS_data, GHTS_data])

elif assay == "CHS":
    dataFiles = glob.glob("/home/gregory.andrews-umw/IBIS/data/" + assay + "/" + TF + "/*" )
    CHS_data = [load_peaks(_) for _ in dataFiles]
    CHS_data = merge_peaks(CHS_data)
    data = CHS_data[["chrom", "start", "end", "abs_summit", "signal"]]
else:
    # assay = GHTS, must provide cycle
    cycle = sys.argv[3]
    prefix += "-" + cycle
    dataFiles = glob.glob("/home/gregory.andrews-umw/IBIS/data/" + assay + "/" + TF + "/*" + cycle + "*")
    data = [load_peaks(_) for _ in dataFiles]
    data = pd.concat(data)

data = data.sort_values(['chrom', 'start'], ascending=[True, True]).reset_index(drop=True)


print(data.head())
# sort like sort -k1,1 -k2,2n


genomeFasta = "/home/gregory.andrews-umw/data/genome/hg38.fa"
genome = Fasta(genomeFasta, as_raw=True, sequence_always_upper=True)


# In[209]:


posSeqs = []
for i in trange(data.shape[0]):
    row = data.iloc[i]
    chrom, start, stop, summit = row.chrom, row.start, row.end, row.abs_summit
    start, stop = int(start), int(stop)
    summit = int(summit)
    seq = genome[chrom][summit-w2:summit+w2]
    posSeqs.append(seq)


# In[210]:


negSeqs =  [dinuclShuffle(_) for _ in posSeqs]


# In[211]:


kmer_w = 6
kmers = find_enriched_gapped_kmers(posSeqs, negSeqs,  3, 0, 18, "dna", False, 8)


# In[212]:


print(kmers)


# In[213]:



# In[214]:


holdOutP = .1
n = len(posSeqs)
weights = data["signal"].tolist()

tmp = list(zip(posSeqs, weights))
random.shuffle(tmp)
posSeqs, weights = map(list, zip(*tmp))
   
posTrainSeqs = posSeqs[:int((1 - holdOutP)*n)]
posTestSeqs = posSeqs[int((1 - holdOutP)*n):]

trainWeights = weights[:int((1 - holdOutP)*n)]
tmp = list(zip(posTrainSeqs, trainWeights))
tmp = sorted(tmp, key = lambda x: x[1])[::-1]
posTrainSeqs, trainWeights = map(list, zip(*tmp))


# In[182]:


print(len(posTrainSeqs), len(trainWeights))


# In[215]:


trainGen = dataGen(posTrainSeqs,
                   padBy=w)
testGen = dataGen(posTestSeqs, 
                  padBy=w,
                  seqsPerEpoch=1000)


# In[216]:


model = construct_model(num_kernels=num_kernels,
                        kernel_width=w,
                        use_bias=use_bias,
                        l1_reg=.00001)


# In[217]:


if use_bias:
    conv_weights, conv_bias = model.get_layer("shared_conv").get_weights()
else:
    conv_weights = model.get_layer("shared_conv").get_weights()[0]


# In[218]:


for i in range(len(kmers)):
    kmer = kmers[i]
    l = len(kmer)
    conv_weights[((w - l)//2):(((w - l)//2)+l),:,i] = encode_sequence(kmer)


# In[219]:


if use_bias:
    model.get_layer("shared_conv").set_weights([conv_weights, conv_bias])
else:
    model.get_layer("shared_conv").set_weights([conv_weights])


# In[220]:


min_lr = .001
max_lr = .1
lr_decay = (min_lr / max_lr) ** (1 / epochs)
schedule = SGDRScheduler(min_lr=min_lr,
                             max_lr=max_lr,
                             steps_per_epoch=trainGen.__len__(),
                             lr_decay=lr_decay,
                             cycle_length=1,
                             mult_factor=1.0, 
                             shape="triangular")

swa = SWA(epochs)


# In[221]:


history = model.fit(trainGen, 
                    steps_per_epoch = trainGen.__len__(), 
                    verbose=2, 
                    epochs=epochs,
                    workers=4,
                    callbacks = [schedule, swa],
                    validation_data = testGen,
                    validation_steps = testGen.__len__())


model.save_weights(prefix + ".h5")


if use_bias:
    conv_weights, conv_bias = model.get_layer("shared_conv").get_weights()
else:
    conv_weights = model.get_layer("shared_conv").get_weights()[0]


# In[223]:


AUCs = []
nSteps = 200
from sklearn.metrics import roc_auc_score
for i in range(num_kernels):
    tmp_conv_weights = np.zeros(conv_weights.shape)
    if use_bias:
        tmp_bias = np.zeros(num_kernels)
    tmp_conv_weights[:,:,i] = conv_weights[:,:,i]
    
    if use_bias:
        tmp_bias[i] = conv_bias[i]
    
    if use_bias:
        model.get_layer("shared_conv").set_weights([tmp_conv_weights, tmp_bias])
    else:
        model.get_layer("shared_conv").set_weights([tmp_conv_weights])
    yPred = model.predict(testGen, steps=nSteps)
    yTest = np.array(nSteps*([1 for i in range(16)] + [0 for i in range(16)]))
    AUCs.append(roc_auc_score(yTest, yPred))
print(AUCs)


# In[224]:


scan_model = construct_scan_model(conv_weights)


# In[225]:


anr = False
thresh = 0
with open(prefix + ".motifs.bed", "w") as f:
    for i in trange(data.shape[0]):
        chrom, start, stop = data.iloc[i][:3]
        start, stop = int(start), int(stop)
        seq = genome[chrom][start:stop]
        
        encoded_seq = np.vstack((0.25*np.ones((w,4)), encode_sequence(seq), 0.25*np.ones((w,4))))
        encoded_seq_rc = encoded_seq[::-1,::-1]

        conv_for = scan_model.predict(np.expand_dims(encoded_seq, axis = 0), verbose=0)[0]
        conv_rc = scan_model.predict(np.expand_dims(encoded_seq_rc, axis = 0), verbose=0)[0]

        for k in range(num_kernels):
            if anr:
                matches_for = np.argwhere(conv_for[:,k] > thresh)[:,0].tolist()
                matches_rc = np.argwhere(conv_rc[:,k] > thresh)[:,0].tolist()
                for x in matches_for:
                    motif_start = x - w 
                    motif_end = motif_start + w
                    score = conv_for[x,k]
                    pfms[k] += encoded_seq[x:x+w,:]

                for x in matches_rc:
                    motif_end = x + w
                    motif_start = motif_end - w 
                    score = conv_rc[x,k] 
                    pfms[k] += encoded_seq_rc[x:x+w,:]
                    n_instances[k] += 1
                
            else:
                maxFor = np.max(conv_for[:,k])
                maxRC = np.max(conv_rc[:,k])

                if maxFor > thresh or maxRC > thresh:
                    if maxFor > maxRC:
                        x = np.argmax(conv_for[:,k])
                        motif_start = x - w 
                        motif_end = motif_start + w
                        score = conv_for[x,k]
                        motifSeq = decode_sequence(encoded_seq[x:x+w,:])
                        print(chrom, start+motif_start, start+motif_end, k, score, "+", motifSeq, file=f, sep="\t")
                    else:
                        x = np.argmax(conv_rc[:,k])
                        motif_end = x + w
                        motif_start = motif_end - w 
                        score = conv_rc[x,k] 
                        motifSeq = decode_sequence(encoded_seq_rc[x:x+w,:])
                        print(chrom, stop-motif_start, stop-motif_start+w, k, score, "-", motifSeq, file=f, sep="\t")


# In[226]:


motifs = pd.read_csv(prefix + ".motifs.bed", sep="\t", names=["chrom", "start", "end", "kernel", "score", "strand", "seq"])
motifs.head()


# In[227]:


motifs = bioframe.closest(motifs, data, suffixes=('_1','_2'))


# In[228]:


motifs.head()

pfms = []
nInstances = []
for i in range(num_kernels):
    tmp = motifs[motifs["kernel_1"] == i]
    if tmp.shape[0] > 0:
        seqs_weights = zip(tmp.seq_1, tmp.signal_2)
        pfm = np.array([encode_sequence(_[0]) *  _[1] for _ in seqs_weights])
        pfm = np.sum(pfm, axis=0)
    else:
        pfm = np.ones([w,4])
    pfms.append(pfm)
    nInstances.append(tmp.shape[0])


# In[231]:


with open(prefix + ".meme", "w") as f:
    print("MEME version 4\n", file=f)
    print("ALPHABET= ACGT\n", file=f)
    print("strands: + -\n", file=f)
    print("Background letter frequencies", file=f)
    print("A 0.25 C 0.25 G 0.25 T 0.25\n", file=f)
    
    for i in range(num_kernels):
        print("MOTIF {}".format(i), file=f)
        print("letter-probability matrix: alength= 4 w= {} nsites= {}".format(w, nInstances[i]), file=f)
        pfm = pfms[i]
        ppm = pfm/pfm.sum(axis=1, keepdims=True)
        for i in range(w):
            print(*ppm[i,:].flatten().tolist(), sep="\t", file=f)
        print(file=f)


# In[232]:


with open(prefix + ".TOMTOM.out", "w") as f:
    run(["tomtom", "--text", "-thresh", "1", prefix + ".meme", "/home/gregory.andrews-umw/IBIS/misc/H12CORE_meme_format.meme"], stdout=f)


# In[233]:


matches = []
pVals = []
for i in range(num_kernels):
    with open(prefix + ".TOMTOM.out") as f:
        for line in f:
            split = line.strip().split("\t")
            try:
                if int(split[0]) == i:
                    matches.append(split[1])
                    pVals.append(float(split[3]))
                    break
            except:
                pass


# In[234]:


fig, axes = plt.subplots(4, 4, figsize=(20,8), tight_layout=True)
toSave = {}
for i in range(num_kernels):
    ax = axes.flatten()[i]
    pfm = pfms[i] + 1
    ppm = pfm/pfm.sum(axis=1, keepdims=True)
    ppm = pd.DataFrame(ppm, columns=["A", "C", "G", "T"])
    trimmed_ppm = trim_ppm(ppm.values, min_info=0.0)[0]
    trimmed_ppm = pd.DataFrame(trimmed_ppm, columns=["A", "C", "G", "T"])
    ic = ppm.applymap(get_information_content)
    
    #ic = trimmed_ppm.map(get_information_content)
    title = "{}\n(p = {:.2e}) N={}; auc={:.3f}".format(matches[i], pVals[i], nInstances[i], AUCs[i])
    logomaker.Logo(ic, ax=ax)
    ax.set_title(title)
    ax.set_ylim([0,2])
    toSave[i] = {"ppm" : ppm.values,
                 "trimmed_ppm" : trimmed_ppm,
                 "weights" : conv_weights[:,:,i],
                 "auc" : AUCs[i],
                 "p" : pVals[i],
                 "match" : matches[i],  
                 "n" : nInstances[i]}
    
    
for i in range(num_kernels,16):
    fig.delaxes(axes.flatten()[i])
    
plt.savefig(prefix + ".png")


# In[236]:


with open(prefix + ".pkl", "wb") as f:
    pickle.dump(toSave, f)    

model.load_weights(prefix + ".h5")
X = np.array([encode_sequence(_) for _ in posSeqs])
y_pred = model.predict(X)
np.save(prefix + ".yPred.npy", y_pred)
