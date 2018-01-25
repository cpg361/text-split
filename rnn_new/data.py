#-*- coding:utf-8 -*-
import logging
import argparse
import os
import os.path
import pprint
#import cPickle as pkl
import pickle as pkl
from collections import Counter
import configurations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--proto",default="get_config_search_gru",
			help="Prototype config to use for config")
parser.add_argument("--ini", default="config.ini",help="configuration file")
args = parser.parse_args()

class PrepareData(object):

	def __init__(self,**kwards):
		
		self.train_file = kwards.pop('train_file')
		self.vocab_train_file =kwards.pop('vocab_train_file')
		self.seq_len = kwards.pop('seq_len')
		self.vocab = kwards.pop('vocab')
		self.vocab_size	= kwards.pop('vocab_size')
		self.unk_token = kwards.pop('unk_token')
		self.bos_token = kwards.pop('bos_token')
		self.eos_token = kwards.pop('eos_token')
		self.unk_id = kwards.pop('unk_id')
		self.bos_id=0
		self.eos_id=0
		self.embed = kwards.pop('embed')
		self.vector = kwards.pop('vector')

		vocab_dict=self._create_dictionary()
		
		print("dictionarys:")
		#for key,val in 	vocab_dict.items():
		#	print key,'\t',val

		with open(self.vocab,'wb') as f:
			pkl.dump(vocab_dict,f,protocol=pkl.HIGHEST_PROTOCOL)
		f.close()
		
		f = open(self.vector,'rb')
		emb  = pkl.load(f)
		f.close()
		id_emb ={}
		for key in vocab_dict.keys():
			if key in emb.keys():
				id_emb[vocab_dict[key]] = emb[key]
			else:
				id_emb[vocab_dict[key]] = emb['unk']
	
		with open(self.embed,'wb') as f:
			pkl.dump(id_emb,f,protocol=pkl.HIGHEST_PROTOCOL)
		f.close()
	def _create_dictionary(self):

		if os.path.isfile(self.train_file):
			ftrain=open(self.train_file,'r')		
		else:	
			logger.error("file [{}] do not exist".format(self.train_file))

		sentences_in = ftrain.readlines()	
		#if os.path.isfile(self.vocab_train_file):
		#	fvocab = open(self.vocab_train_file,'r')
		#else:
	        #		logger.error("file [{}] do not exist".format(self.vocab_train_file))
		 
		ftrain.close()
		#sentences_vocab = fvocab.readlines()
		#fvocab.close()
		counter_word= Counter()
#		counter_vocab = Counter()

		for line in sentences_in:
			words = line.strip().split()
			

			if len(words)==0:
				continue
			if self.seq_len <len(words):
				continue
			
			counter_word.update(words)
		#for line in sentences_vocab:
		#	words = line.strip().split()
		#	
		#	if len(words)==0:
		#		continue
		#	if self.seq_len < len(words):
		#		continue
		#	counter_word.update(words)
#
		logger.info("Source Total: %d unique words, with a total of %d words." 
				%(len(counter_word),sum(counter_word.values()))	)

		special_tokens=[self.unk_token,self.bos_token,self.eos_token]
		for st in special_tokens:
			if st in counter_word:
				del counter_word[st]
#			if st in counter_vocab:
#				del counter_vocab[st]

		if self.vocab_size <2 :
			self.vocab_size = len(counter_word)+2

		valid_count = counter_word.most_common(self.vocab_size-2)
#		val_count = counter_vocab.most_common(self.vocab_size-len(valid_count)-2)
		vocab_dict ={self.bos_token:self.bos_id,self.eos_token:self.eos_id,self.unk_token:self.unk_id}
		vocab_word_counts = 0
		k=1
		for i,(word,count) in enumerate(valid_count):
			vocab_dict[word] = i + 2
			k=k+1
			vocab_word_counts += count
#		for i,(word,count) in enumerate(val_count):
#			vocab_dict[word] = i + k
		logger.info(' dict contains %d words,covering %.1f%% of the text' %(self.vocab_size,100.0*vocab_word_counts/sum(counter_word.values())))

		return vocab_dict


if __name__ == '__main__':
	if not os.path.isfile(args.ini):
		raise Exception("File not found:" + args.ini)
	
	configuration = getattr(configurations,args.proto)(args.ini)
	logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))
	PrepareData(**configuration)
