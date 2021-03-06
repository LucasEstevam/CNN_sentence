import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import pdb
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()


def load_bin_vec(fname, vocab):
  """
  Loads 300x1 word vecs from Google (Mikolov) word2vec
  """
  word_vecs = {}
  with open(fname, "rb") as f:
      header = f.readline()
      vocab_size, layer1_size = map(int, header.split())
      binary_len = np.dtype('float32').itemsize * layer1_size
      for line in xrange(vocab_size):
          word = []
          while True:
              ch = f.read(1)
              if ch == ' ':
                  word = ''.join(word)
                  break
              if ch != '\n':
                  word.append(ch)   
          if word in vocab:
             word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
          else:
              f.read(binary_len)
  return word_vecs

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()


def load_bin_vec(fname, vocab):
  """
  Loads 300x1 word vecs from Google (Mikolov) word2vec
  """
  word_vecs = {}
  with open(fname, "rb") as f:
      header = f.readline()
      vocab_size, layer1_size = map(int, header.split())
      binary_len = np.dtype('float32').itemsize * layer1_size
      for line in xrange(vocab_size):
          word = []
          while True:
              ch = f.read(1)
              if ch == ' ':
                  word = ''.join(word)
                  break
              if ch != '\n':
                  word.append(ch)   
          if word in vocab:
             word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
          else:
              f.read(binary_len)
  return word_vecs


def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]     

if __name__=="__main__":
    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
        
    non_static=False
    execfile("conv_net_classes.py")    
    U = W
    
    savedparams = cPickle.load(open('classifier.save','rb'))

    filter_hs=[3,4,5]
    conv_non_linear="relu"
    hidden_units=[100,2]
    dropout_rate=[0.5]
    activations=[Iden]
    img_h = 56 + 4 + 4
    img_w = 300
    rng = np.random.RandomState(3435)
    batch_size=50
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))

#define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))],allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    classifier.params[0].set_value(savedparams[0])
    classifier.params[1].set_value(savedparams[1])
    k = 2
    for conv_layer in conv_layers:
        conv_layer.params[0].set_value( savedparams[k])
        conv_layer.params[1].set_value( savedparams[k+1])
        k = k + 2

    datasets = make_idx_data_cv(revs, word_idx_map, 1, max_l=56,k=300, filter_h=5)
    test_set_x = datasets[1][:,:img_h] 
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    test_pred_layers = []
    test_size = 1
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict_p(test_layer1_input)
    #test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x],test_y_pred,allow_input_downcast=True)   
    
    #test_loss = test_model_all(test_set_x,test_set_y) 
    #test_perf = 1- test_loss   
    #print test_perf
    w2v_file = "word2vec.bin"

    line = "this is terrible."

    rev = []
    rev.append(line.strip())
    orig_rev = clean_str(" ".join(rev))
    datum  = [{"y":1, 
              "text": orig_rev,                             
              "num_words": len(orig_rev.split())}]
    sent = get_idx_from_sent(orig_rev, word_idx_map, 56, k, filter_h)   
    #yvalue
    sent.append(1)
    test = np.array([sent],dtype="int")
    test_set_x = test[:,:img_h] 
    test_set_y = np.asarray(test[:,-1],"int32")

    test_loss = test_model_all(test_set_x) 
    #test_perf = 1- test_loss   
    print test_loss
