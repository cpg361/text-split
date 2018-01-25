import logging
#import cPickle as pkl
import pickle as pkl
import os
import argparse
import numpy
import gensim
#from fuel.datasets import TextFile
#from fuel.streams import DataStream

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

numpy.set_printoptions(suppress=True)
numpy.set_printoptions(threshold='nan')
class DStream(object):

    def __init__(self,**kwards):
    
        self.vocab = kwards.pop('vocab')
        self.unk_id = kwards.pop('unk_id')
        self.unk_token = kwards.pop('unk_token')
        self.train_file_with_label = kwards.pop('train_file_with_label')
        self.train_file = kwards.pop('train_file')
        self.eos_token = kwards.pop('eos_token')
        self.seq_len = kwards.pop('seq_len')        
        self.batch_size = kwards.pop('batch_size')
        self.sort_k_batches = kwards.pop('sort_k_batches')
        self.embed_num = kwards.pop('embed_num')
        self.vocab_dict = self._get_dict()
        self.eos_id =self.vocab_dict[self.eos_token]
        self.model = gensim.models.Word2Vec.load('dic_18_hlt.bin')
        self.sentence ,self.sentence_label= self._get_sentence_pairs()
        assert len(self.sentence)==len(self.sentence_label)

        if self.sort_k_batches > 1:
            self.sentence,self.sentence_label=self._sort_by_k_batches(self.sentence,self.sentence_label)
        
        num_sents=len(self.sentence)
        if num_sents % self.batch_size == 0:
            self.blocks = num_sents // self.batch_size
        else:
            self.blocks = num_sents // self.batch_size + 1
    
    def get_iterator(self):
    
        for i in range(self.blocks):
            lines = self.sentence[i*self.batch_size: (i+1)*self.batch_size]
            x=[]
            for line in lines:
                word_list=[]
                for word in line:
                    if self.model.__contains__(word):
                        wv = self.model.__getitem__(word)
                        vector = [v for v in wv]
                    else:
                        wv = self.model.__getitem__('unk')
                        vector = [v for v in wv]
                    word_list.append(vector)
                x.append(word_list)
            x_label =self.sentence_label[i*self.batch_size: (i+1)*self.batch_size]
            batch = self._create_padded_batch(x, x_label)
            yield batch
    
    def _create_padded_batch(self, x ,x_label):

        mx = self.seq_len        
        batch_size= len(x)
        wv_size = self.embed_num
        X=numpy.zeros((batch_size, mx, wv_size), dtype ='float32')
        X_label = numpy.zeros((batch_size,mx), dtype = 'int64')
        Xmask = numpy.zeros((batch_size,mx), dtype = 'float32')
        for idx in range(len(x)):
            X[idx,:len(x[idx])] = x[idx]
            Xmask[idx,:len(x[idx])] = 1.
            X_label[idx,:len(x_label[idx])] = x_label[idx]
            if len(x[idx])< mx:
                #X[idx,len(x[idx]):]=self.eos_id
                X_label[idx,len(x[idx]):] =self.eos_id
                Xmask[idx,len(x[idx])] = 1.
        return X, X_label, Xmask
    

    def _sort_by_k_batches(self,sentence,sentence_label):
        
        bs = self.batch_size * self.sort_k_batches
        num_sents=len(sentence)
        if num_sents % bs == 0:
            blocks = num_sents // bs
        else:
            blocks = num_sents // bs + 1
        sort_sentence = []
        sort_sentence_label = []
        for i in range(blocks):
            tmp_sentence = numpy.asarray(sentence[i*bs:(i+1)*bs])
            tmp_sentence_label = numpy.asarray(sentence_label[i*bs:(i+1)*bs])
            lens = numpy.asarray([list(map(len,tmp_sentence))])
            orders = numpy.argsort(lens[-1])
            for idx in list(orders):
                sort_sentence.append(tmp_sentence[idx])
                sort_sentence_label.append(tmp_sentence_label[idx])
        return sort_sentence, sort_sentence_label

    def _get_sentence_pairs(self):
    
        if os.path.isfile(self.train_file):
            fin = open(self.train_file,'r')
        else:
            logger.error("file [{}] do not exist".format(self.train_file))
    
        sentences =[]
        sentences_label=[] 

        if os.path.isfile(self.train_file_with_label):
            fin_label = open(self.train_file_with_label,'r')
        else:
            logger.error("file [{}] do not exist".format(self.train_file_with_label))
            
        for line,line_label in zip(fin,fin_label):
            words = line.strip().split()
            words_lab = line_label.strip().split()
            word_label=[]
            if len(words)==0 :
                continue
            if len(words)> self.seq_len:
                continue
            #word_ids = [self.vocab_dict[w] if w in self.vocab_dict else self.unk_id for w in words ]    
            sentences.append(words)
            for l in range(len(words_lab)-1):
                if words_lab[l]=="|||":
                    continue
                if words_lab[l]!='|||' and words_lab[l+1]=="|||":
                    word_label.append(1)
                else:   
                    word_label.append(0)
            sentences_label.append(word_label)

        fin.close()
        fin_label.close()
        
        return sentences , sentences_label
            


    def _get_dict(self):
        if os.path.isfile(self.vocab):
            vocab_dict =pkl.load(open(self.vocab,'rb'))    
        else:    
            logger.error("file [{}] do not exist".format(self.vocab))
        return vocab_dict

def get_devtest_stream(data_type='valid',input_file=None,**kwards):
    
    model = gensim.models.Word2Vec.load('dic_18_hlt.bin')
    if data_type == 'valid':
        data_file = kwards.pop('valid_file')
    elif data_type == 'test':
        if input_file is None:
            data_file = kwards.pop('test_file')
        else:
            data_file = input_file
    else:
        logger.error('wrong datatype, which must be one of valid or test')
    
    unk_token = kwards.pop('unk_token')
    eos_token = kwards.pop('eos_token')
    vocab = kwards.pop('vocab')
    embed_num = kwards.pop('embed_num')
    
    dictionary=pkl.load(open(vocab,'rb'))
    fin=open(data_file,'rU')    
    dev_stream=[]
    for line in fin:
        words=line.strip().split()    
        words.append('</S>')
        word_list=[]
        for word in words:
            if model.__contains__(word):
                wv = model.__getitem__(word)
                vector = [v for v in wv]
            else:
                vector =[0.0]*embed_num
            word_list.append(vector)
        #dev_id=[dictionary[w] if w in dictionary else 1 for w in word]
        dev_stream.append(word_list)
    
    
        
    return dev_stream

if  __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto",default="get_config_search_gru",
                help="Prototype config to use for config")
    parser.add_argument("--ini",default="config.ini",
                help="configure file")
    args = parser.parse_args()

    import configurations
    configuration =getattr(configurations,args.proto)(args.ini)
    ds=DStream(**configuration)
    i=0
    for x,x_label,x_mask in ds.get_iterator():
        i = i+1
        print(i)
        #print(x.shape)
        #print(x_label[0])
        #print(x_mask.shape)
        #print('')










