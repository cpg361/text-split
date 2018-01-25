import numpy
import argparse
import configurations
from train import Encoder,Test
#from stream import get_devtest_stream
import tensorflow as tf
#import cPickle as pkl
import pickle as pkl
from hlt import HLT

class Benebot_textsplit():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--proto",default="get_config_search_gru",
					help="Prototype config to use for config")
        #parser.add_argument('test_file',type=str)
        args = parser.parse_args()
        self.configuration = getattr(configurations,args.proto)()
        
        self.encoding=Encoder(**self.configuration)
        self.encoding.build_sampler()
        sample_fn = self.encoding.step_sample
        self.testing=Test(self.encoding,**self.configuration)
        self.vocab = self.configuration.pop('vocab')
        self.dictionary,self.id_dictionary = self.testing._get_dict(self.vocab)
        self.ss = HLT()
        self.ss.get_web_content()
        self.gpu_options = tf.GPUOptions(visible_device_list='1',allow_growth=True)

    def get_split_sent(self,sent,hlt_flag=True):
        if hlt_flag:
            sent = self.ss.get_ws(sent)
        sent_list = sent.strip().split()
        sent_list.append('</S>')
        sent_id = [self.dictionary[w] if w in self.dictionary else 1 for w in sent_list]
        sent_mask = [0. if unit is 0 else 1. for unit in sent_id]
        prob=self.encoding.step_sample(sess,(numpy.array(sent_id).flatten().T)[:,None],(numpy.array(sent_mask).flatten())[:,None])
        order = numpy.argmax(prob,axis=-1)
        order = order.flatten()
        sent_id = (numpy.array(sent_id).T).flatten()
        res = self.testing._idx_to_word(sent_id[:],self.id_dictionary)
        words_list = sent_list[:-1]
        
        order = order[:-1]
        string = ''
        for word_unit,order_unit in zip(words_list,order):
            if order_unit == 1:
                string = string+word_unit+' / '
            else:
                string = string+word_unit+' '
        if string.endswith(' / '):
            string = string[:-3]
        return string
        

if __name__=='__main__':

    ts = Benebot_textsplit()


    with tf.Session(config=tf.ConfigProto(gpu_options=ts.gpu_options,allow_soft_placement=True)) as sess:
        ts.encoding.load(session=sess,path=ts.configuration['saveto_split_best'])
        #ts = get_devtest_stream(data_type='test',input_file=args.test_file,**configuration)
        while 1:
            sent = input('Enter a sentence:')
            result = ts.get_split_sent(sent,True)
            print(result)
