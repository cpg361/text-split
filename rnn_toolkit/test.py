import numpy
import argparse
import logging
import configurations
import pprint
import time
import os
import os.path
from train import Encoder,Test
from stream import get_devtest_stream
import tensorflow as tf
import cPickle as pkl
from hlt import HLT

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--proto",default="get_config_search_gru",
					help="Prototype config to use for config")
    #parser.add_argument('test_file',type=str)
    args = parser.parse_args()
    configuration = getattr(configurations,args.proto)()
    encoding=Encoder(**configuration)
    encoding.build_sampler()
    sample_fn = encoding.step_sample
    testing=Test(encoding,**configuration)
    #ss=HLT()
    gpu_options = tf.GPUOptions(visible_device_list='1',allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        encoding.load(session=sess,path=configuration['saveto_split_best'])
        vocab = configuration.pop('vocab')
        dictionary,id_dictionary = testing._get_dict(vocab)
        #ts = get_devtest_stream(data_type='test',input_file=args.test_file,**configuration)
        while 1:
            sent = raw_input('Enter a sentence:')
            #sent = ss.get_ws(sent)
            id_list = []
            sent_list = sent.strip().split()
            sent_list.append('</S>')
            sent_id = [dictionary[w] if w in dictionary else 1 for w in sent_list]
            id_list.append(sent_id)
            sent_mask = [0. if unit is 0 else 1. for unit in sent_id]
            prob=encoding.step_sample(sess,(numpy.array(sent_id).flatten().T)[:,None],(numpy.array(sent_mask).flatten())[:,None])
            order = numpy.argmax(prob,axis=-1)
            order = order.flatten()
            sent_id = (numpy.array(sent_id).T).flatten()
            res = testing._idx_to_word(sent_id[:],id_dictionary)
            words_list = res.strip('</S>').split()
            order = order[:-1]
            string = ''
            for word_unit,order_unit in zip(words_list,order):
                if order_unit == 1:
                    string = string+word_unit+' / '
                else:
                    string = string+word_unit+' '
            print string
            print order
            #testing.apply(sess,ts,configuration['test_out'])
