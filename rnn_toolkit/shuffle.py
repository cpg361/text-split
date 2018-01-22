import argparse
import numpy

def RunShuffle(train_file,train_file_with_label,train_file_shuffle,train_file_with_label_shuffle):
	
	ftrain_file = open(train_file,'rU')
	ftrain_file_with_label = open(train_file_with_label,'rU')

	wtrain_file = open(train_file_shuffle,'w')
	wtrain_file_with_label = open(train_file_with_label_shuffle,'w')

	sent_in = []
	sent_in_with_label=[]
	
	for line in ftrain_file:
		sent_in.append(line.strip())
	for line in ftrain_file_with_label:
		sent_in_with_label.append(line.strip())

	assert len(sent_in) == len(sent_in_with_label)
	
	idxs= numpy.arange(len(sent_in))
	numpy.random.shuffle(idxs)
	
	for i in idxs:
		wtrain_file.write(sent_in[i]+'\n')
		wtrain_file_with_label.write(sent_in_with_label[i]+'\n')
	ftrain_file.close()
	ftrain_file_with_label.close()
	wtrain_file.close()
	wtrain_file_with_label.close()
		
		
if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument("train_file",type=str)
	parser.add_argument("train_file_with_label",type=str)
	args = parser.parse_args()

	RunShuffle(args.train_file,args.train_file_with_label,args.train_file+'.shuffle',args.train_file_with_label+'.shuffle')
