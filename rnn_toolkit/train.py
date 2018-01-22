from __future__ import division
import numpy
import tensorflow as tf
import argparse
#import cPickle as pkl
import pickle as pkl
import logging
import pprint
import time
from stream import DStream,get_devtest_stream
from initialization import constant_weight, uniform_weight, ortho_weight, normal_weight
import configurations
import os
import sys
from deal_back import eval_method

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _p(pp, name):
    return '%s_%s' % (pp, name)

class Encoder(object):
    
    def __init__(self,**kwargs):
        self.nhids_num = kwargs.pop('nhids_num')
        self.embed_num = kwargs.pop('embed_num')
        self.vocab_size = kwargs.pop('vocab_size')
        self.dropout = kwargs.pop('dropout')
        self.seq_len = kwargs.pop('seq_len')
        self.path = kwargs.pop('saveto_split')
        self.emb = kwargs.pop('embed')
            
        with tf.variable_scope("TABLE"):
            self.table= LookupTable(self.vocab_size,self.embed_num,self.emb,name='table')
                    
    
        with tf.variable_scope("BIRNN_ENCODER"):
            self.encoder = BidirectionalEncoder(self.embed_num,self.nhids_num, self.table, name = 'birnn_encoder')
        
        self.n_out=2
        with tf.variable_scope("LOGISTIC"):
            self.logistic_layer = LogisticRegression(self.embed_num,self.n_out)
        self.saver = tf.train.Saver()
        self.saver_best = tf.train.Saver()

    def nmt(self,src,src_mask,src_label) :
        tf_sents=tf.transpose(src,perm=[1,0])
        tf_sents_mask = tf.transpose(src_mask,perm=[1,0])
        tf_sents_label =  tf.transpose(src_label,perm=[1,0])

        readout = self.encoder.apply(tf_sents,tf_sents_mask)    
        if self.dropout < 1.0:
            readout=tf.nn.dropout(readout, keep_prob=self.dropout)
             #   p= self.logistic_layer.get_probs(readout,tf_sents_mask)
                    
        self.cost = self.logistic_layer.cost(readout,tf_sents_label,tf_sents_mask)

        self.L1=tf.add_n(tf.get_collection('L1_losses'))
        self.L2=tf.add_n(tf.get_collection('L2_losses'))

        train_cost = self.cost +self.L1+self.L2
        
        return train_cost
    
    
    def build_trainer(self):
        self.src = tf.placeholder(dtype=tf.int32,shape=(None,self.seq_len),name="src_holder_train")
        self.src_mask = tf.placeholder(dtype=tf.float32,shape=(None,self.seq_len),name="src_mask_holder_train")
        self.src_label = tf.placeholder(dtype=tf.int32,shape=(None,self.seq_len),name="src_label_holder_train")
                
        self.train_cost = self.nmt(self.src,self.src_mask,self.src_label)

        params = tf.trainable_variables()
        opt=tf.train.AdadeltaOptimizer(0.1,rho=0.95,epsilon=1e-6)    
        grads = tf.gradients(self.train_cost,params)

        clipped_gradients,norm =tf.clip_by_global_norm(grads,1.0)     
        self.train_op = opt.apply_gradients(zip(clipped_gradients,params))


    def step_train(self,session,sents,sents_label,sents_mask):
        inps = {self.src:sents,self.src_mask : sents_mask , self.src_label :sents_label}
        outs= [self.train_cost,self.train_op]
        outputs = session.run(outs,inps)

        return outputs


    def build_sampler(self):

        self.src_sample=tf.placeholder(dtype=tf.int32, shape=(None,1),name="src_holder_sample")
        self.src_mask_sample=tf.placeholder(dtype=tf.float32,shape=(None,1),name="src_mask_sample")
                    
        readout = self.encoder.apply(self.src_sample,self.src_mask_sample)
        
        if self.dropout < 1.0:
            readout = tf.nn.dropout(readout, keep_prob=self.dropout)
        self.prob = self.logistic_layer.get_probs(readout,self.src_mask_sample)

    def step_sample(self,session,src_sample,src_mask_sample):
        inps_sample = {self.src_sample:src_sample,self.src_mask_sample:src_mask_sample}
        outs_sample=[self.prob]
        sample_probs = session.run(outs_sample,inps_sample)
        
        return sample_probs

    def save(self, session,iters=0,model_name="train_model.ckpt",path=None,best_flag=None):
        if path is None:
            path = self.path
        checkpoint_path = os.path.join(path,model_name)
        if not best_flag:
            self.saver.save(session,checkpoint_path,global_step=iters)
        else:
            self.saver_best.save(session,checkpoint_path,global_step=iters)

    
    def  load(self,session,path=None):
        if path is None:
            path = self.path
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            session.run(tf.global_variables_initializer())
            logger.info("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
            self.saver.restore(session,ckpt.model_checkpoint_path)
        else:
            logger.info("model file not exist")
            init =tf.global_variables_initializer()
            session.run(init)


class  LogisticRegression(object):

    def __init__(self,n_in,n_out,name='LR'):
        self.name=name
        self.n_in=n_in
        self.n_out=n_out
        self.W = normal_weight(shape=(n_in, n_out), name=_p(name, 'W'))

             
        self.b = constant_weight(shape=(n_out, ), name=_p(name, 'b'))


    def get_probs(self, input,mask):
                
        #print mask.get_shape().ndims
        
        seq_len = mask.shape[0]
            
    #    if seq_len is None:
    #        seq_len = 1
    #    print seq_len
        input_2d = tf.reshape(input,[-1,self.n_in])
        energy =tf.matmul(input_2d, self.W) + self.b
#        mask_flat = tf.reshape(mask,[-1])
    #    energy=tf.multiply(energy,mask_flat[:,None])
 #           energy=tf.reshape(energy,[seq_len,-1,self.n_out])
        p_y_given_x = tf.nn.softmax(energy,-1)

        p_y_given_x = tf.reshape(p_y_given_x,[-1,self.n_out])
        return p_y_given_x


    def cost(self, p_y_given_x, targets, mask):
        prediction = p_y_given_x
        input_2d = tf.reshape(prediction,[-1,self.n_in])

        energy =tf.matmul(input_2d, self.W) + self.b

        targets_flat_sample=tf.reshape(targets,[-1])
        mask_flat = tf.reshape(mask,[-1])
        ce =tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets_flat_sample,logits=energy,name=_p(self.name,'sparse_cross_entropy_sample'))
        ce_new=tf.reshape(ce,[targets.shape[0],-1])
                 
        return tf.reduce_mean(tf.reshape(tf.reduce_sum(ce_new,0),[-1]))

class  BidirectionalEncoder(object):
    def __init__(self, n_in,n_hids,table,name = 'rnn_encoder'):
        self.pname = name
        self.table = table
        self.n_in = n_in
        self.n_hids = n_hids

        self.n_cdim = self.n_hids * 2
        self.W_o_h = normal_weight(shape=(self.n_cdim,self.n_in),name=_p(self.pname,'W_o_h'))
        self.W_o_e = normal_weight(shape=(self.n_in,self.n_in),name=_p(self.pname,'W_o_e'))
        self.b_o = constant_weight(shape=(self.n_in,),name=_p(self.pname,'b_out_o'))

        self.forward = GRU(self.n_in,self.n_hids,name=_p(name,'forward'))
    
        self.backward = GRU(self.n_in,self.n_hids,name=_p(name,'backward'))

    def _step_readout(self,readout_1,x):
        state_below_i,hiddens_i=x[0],x[1]
        readout_out=tf.matmul(hiddens_i,self.W_o_h) + tf.matmul(state_below_i,self.W_o_e)+self.b_o             
        return readout_out
        
    def readout(self,state_below,hiddens):
            
        sequences = (state_below,hiddens)
        rval = tf.scan(fn=self._step_readout,elems=sequences,initializer = tf.matmul(state_below[0,:,:],tf.zeros([self.n_in,self.n_in])))

        return rval



    def apply(self,sentence,sentence_mask):

        state_below =self.table.apply(sentence)
        hiddens_forward = self.forward.apply(state_below,sentence_mask)

        back_state_below =tf.reverse(state_below,[0])
        if sentence_mask is None:
            hiddens_backward =self.backward.apply(back_state_below)
        else:
            back_state_mask =tf.reverse(sentence_mask,[0])
            hiddens_backward = self.backward.apply(back_state_below,back_state_mask)
            hiddens_backward = tf.reverse(hiddens_backward,[0])
        annotaitons =tf.concat([hiddens_forward,hiddens_backward],state_below.get_shape().ndims-1)
        readout =self.readout(state_below,annotaitons)
                
        return readout 


class  LookupTable(object):
        
    def __init__(self,vocab_size,embed_num,embedding_file, name='embeddings'):

        self.vocab_size = vocab_size
        self.embedding_size =embed_num
        self.embedding_file = embedding_file
        self.name=name        
        if os.path.isfile(self.embedding_file):
            f=open(self.embedding_file,'rb')
        else:
            logger.error("file do not exist")
        emb_dic = pkl.load(f)
        f.close()
        
        #emb = emb_dic[0].reshape(1, -1)
        #for i in range(1, len(emb_dic)):
        #    emb = numpy.row_stack((emb, emb_dic[i].reshape(1, -1)))
        emb = numpy.array(list(emb_dic.values()))
        if len(emb_dic) < vocab_size:
            for i in range(len(emb_dic), vocab_size):
                emb = numpy.row_stack((emb, emb_dic[1].reshape(1, -1)))
        assert len(emb) == vocab_size


        len_list=len(emb[0])    
        #self.W=normal_weight([self.vocab_size,self.embedding_size],name=self.name)
        self.W=tf.get_variable(name=name,initializer=emb)

    def apply(self,indices):

        input_embedding=tf.nn.embedding_lookup(self.W,indices)

        return input_embedding    


class GRU(object):
    
    def __init__(self, n_in ,n_hids,name='GRU'):
        self.n_in = n_in
        self.n_hids = n_hids
        self.pname = name
        
        self._init_params()
        
    def _init_params(self):

        shape_xh = (self.n_in,self.n_hids)
        shape_hh = (self.n_hids,self.n_hids)
        
        self.W_xz = normal_weight(shape=shape_xh,name=_p(self.pname,'W_xz'))
        self.W_xr = normal_weight(shape=shape_xh,name=_p(self.pname,'W_xr'))
        self.W_xh = normal_weight(shape=shape_xh,name=_p(self.pname,'W_xh'))
        self.b_z = constant_weight(shape=(self.n_hids, ),name=_p(self.pname,'b_z'))
        self.b_r = constant_weight(shape=(self.n_hids, ),name=_p(self.pname,'b_r'))
        self.b_h = constant_weight(shape=(self.n_hids, ),name=_p(self.pname,'b_h'))
        self.W_hz = ortho_weight(shape=shape_hh,name=_p(self.pname,'W_hz'))
        self.W_hr = ortho_weight(shape=shape_hh,name=_p(self.pname,'W_hr'))
        self.W_hh = ortho_weight(shape=shape_hh,name=_p(self.pname,'W_hh'))
                
    
    def _step(self,h_tm1,x):
            
        x_state,x_m=x[0],x[1]
        z_t = tf.sigmoid(tf.matmul(x_state,self.W_xz) + tf.matmul(h_tm1,self.W_hz)+self.b_z)
        r_t = tf.sigmoid(tf.matmul(x_state,self.W_xr) + tf.matmul(h_tm1,self.W_hr)+self.b_r)
        can_h_t = tf.tanh(tf.matmul(x_state,self.W_xh) + r_t*tf.matmul(h_tm1,self.W_hh)+self.b_h)

        h_t = tf.multiply((1. - z_t), h_tm1) +tf.multiply(z_t,can_h_t)
        h_t =tf.multiply(x_m[:,None],h_t) + tf.multiply((1. - x_m[:,None]) , h_tm1)
        
        return h_t

    def apply(self,state_below,mask_below=None,init_state=None):
        n_steps =state_below.shape[0]
        if state_below.get_shape().ndims == 3:
            batch_size = state_below.shape[1]
        else:
            batch_size=1
            state_below = tf.reshape(state_below,(n_steps,batch_size,-1))
        
        if init_state is None:
            init_state = tf.matmul(state_below[0,:,:],tf.zeros([self.n_in,self.n_hids]))
                
        sequences=(state_below,mask_below)    
        rval = tf.scan(fn=self._step,elems=sequences,initializer=init_state)
        
        self.output =rval
        return self.output
    
class Sampler(object):
    
    def __init__(self,sample_fn,**kwards):
        self.sample_fn=sample_fn
        self.unk_token = kwards.pop('unk_token')
        self.vocab = kwards.pop('vocab')
        self.eos_token = kwards.pop('eos_token')
        self.hook_samples = kwards.pop('hook_samples')
        self.dict_vocab ,self.iddict_vocab= self._get_dict(self.vocab)

    def _get_dict(self,vocab_file):
        if os.path.isfile(vocab_file):
            ddict = pkl.load(open(vocab_file,'rb'))
        else:  
            logger.error("file [{}] do not exist".format(vocab_file))

        iddict = dict()
        for kk,vv in ddict.items():
            iddict[vv] = kk
        iddict[0] = self.eos_token
        
        return ddict,iddict
            
    def _get_true_length(self,seq,vocab):
        try:
            return seq.tolist().index(vocab[self.eos_token])+1
        except ValueError:
            return len(seq)    
    
    def _idx_to_word(self,seq,ivocab):
        return " ".join([ivocab.get(idx,self.unk_token) for idx in seq])

    def apply(self,session,sent_batch,sent_mask_batch,sent_label_batch):
        batch_size = sent_batch.shape[0]
        hook_samples  = min(batch_size,self.hook_samples)
        sample_idx = numpy.random.choice(batch_size,hook_samples,replace=False)
        input_ = sent_batch[sample_idx,:]
        input_mask = sent_mask_batch[sample_idx,:]
        label_target = sent_label_batch[sample_idx,:]
        
        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i],self.dict_vocab)
            target_label = label_target[i]
            inp = input_[i,:]
            inp_mask = input_mask[i,:]
            prob=self.sample_fn(session,inp[:,None],inp_mask[:,None])
                        
            logger.info("Input:{}".format(self._idx_to_word(input_[i][:],self.iddict_vocab)))
            #logger.info("target:{}".format(str(target_label)))
            order=numpy.argmax(prob,axis = -1)
            order=order.flatten()
            #logger.info("Output:{}".format(order))
            error=(abs(target_label - order)).sum()
            cost = error/len(target_label)
            #logger.info("eval: {}".format(cost))

class Test(object):

    def __init__(self,encoder, test_file = None , **kwards):
        self.encoder = encoder    
        self.unk_token = kwards.pop('unk_token')
        self.vocab = kwards.pop('vocab')
        self.eos_token = kwards.pop('eos_token')
        if test_file is None:
            self.test_file = kwards.pop('valid_file')

        self.dict_vocab ,self.iddict_vocab = self._get_dict(self.vocab)

    def apply(self,session,data_stream,out_file):
        logger.info("Begin split ...")
        fout = open(out_file,'w')
        val_start_time = time.time()
        i=0
        for sent in data_stream:
            i=i+1
            out_split=[]
            sent_mask = [0. if unit is 0 else  1. for unit in sent]
            prob=self.encoder.step_sample(session,(numpy.array(sent).flatten().T)[:,None],(numpy.array(sent_mask).flatten())[:,None])
            order=numpy.argmax(prob,axis = -1)
            order=order.flatten()
            sent = (numpy.array(sent).T).flatten()
            res = self._idx_to_word(sent[:],self.iddict_vocab)
            words=res.strip('</S>').split()
            order = order[:-1]
            assert len(words)==(len(order))
            string =""
            for words_unit,order_unit in zip(words,order):
                if order_unit == 1:
                    string=string+words_unit+' ||| '
                else:
                    string =string+words_unit+' '
                       # print string
            fout.write(string.strip(' |||')+' |||')
            fout.write('\n')
            if i % 100 ==0:
                logger.info("Split {} lines of valid/test set ..".format(i))
        fout.close()    

    def _get_dict(self,vocab_file):

        if os.path.isfile(vocab_file):
            ddict = pkl.load(open(vocab_file,'rb'))
        else:
            logger.error("file [{}] do not exist".format(vocab_file))

        iddict = dict()
        for kk,vv in ddict.items():
            iddict[vv] = kk

        iddict[0] = self.eos_token

        return ddict,iddict

    def _idx_to_word(self,seq,ivocab):
        return " ".join([ivocab.get(idx,self.unk_token) for idx in seq])
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", default="get_config_search_gru",help="Prototype config to use for config")
#    parser.add_argument("--ini", default="config.ini",help="configuration.ini")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    
    configuration = getattr(configurations,args.proto)()
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))
    

    encoding = Encoder(**configuration)
    encoding.build_trainer()
    train_fn = encoding.step_train
    encoding.build_sampler()
    sample_fn = encoding.step_sample

    ds = DStream(**configuration)
    vs = get_devtest_stream(data_type='valid',input_file=None,**configuration)

    max_epochs = configuration['finish_after']
    
    testing = Test(encoding,**configuration)
#    sampler = Sampler(encoding,**configuration)
    
    iters = 0
    best_F1 = -1
    best_P= -1
    best_R= -1
    gpu_options = tf.GPUOptions(visible_device_list='1',allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        if configuration['reload']:
                encoding.load(sess)
        sampler = Sampler(sample_fn=sample_fn,**configuration)

        for epoch in range(max_epochs):
            print('epoch:',epoch)
            for x ,x_label,x_mask in ds.get_iterator():
                last_time = time.time()
                tc = train_fn(sess,x,x_label,x_mask)
                cur_time=time.time()
                iters+=1
                logger.info('epoch %d \t updates %d train cost %4f use time %4f'%(epoch,iters,tc[0],cur_time-last_time))
                if iters% configuration['sample_freq'] == 0:
                    sampler.apply(sess,x,x_mask,x_label)

                if iters% configuration['save_freq'] == 0:
                    encoding.save(session=sess,iters=iters,model_name="train_mid_model"+str(iters)+'.ckpt',path=configuration['saveto_split'])
            
                    testing.apply(sess,vs,configuration['valid_out'])
                    P,R,F1=eval_method(configuration['valid_out'],configuration['valid_split'])
                    logger.info('valid_test \t epoch %d \t updates %d P %.4f,R %.4f,F1_sum %.4f'%(epoch,iters,P,R,F1))
                
                    if F1 > best_F1 :
                        best_epoch = epoch
                        best_iters = iters
                        best_F1 = F1
                        best_P=P
                        best_R=R
                        encoding.save(session=sess,iters=iters,model_name='train_best_model.ckpt',path=configuration['saveto_split_best'],best_flag=True)
                    logger.info('best_valid_test \t epoch %d \t updates %d P %.4f R %.4f  F1 %.4f'%(best_epoch,best_iters,best_P,best_R,best_F1))
