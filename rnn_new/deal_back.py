#coding:utf-8
from __future__ import division
import argparse
import os
import os.path

def eval_method(file_in,file_ref):
    
    sentence_in=[]
    sentence_ref=[]
    with open(file_in,'rU') as fin:
        for line in fin:
            line = line.strip()
            sentence_in.append(line)
    fin.close()
    with open(file_ref,'rU') as fref:
        for line in fref:
            line = line.strip()
            sentence_ref.append(line)

    assert len(sentence_ref) == len(sentence_in)
    fref.close()
    
    i=0
    F1_sum=0
    sum_count=0
    sum_count_in= 0
    sum_count_ref= 0
    for line_ref ,line_in in zip(sentence_ref,sentence_in):
        count_in =0
        count_ref = 0
        pos_ref = 0
        ref_pos=[]
        word_ref=line_ref.strip().split()
        word_in = line_in.strip().split()
        for unit_ref in word_ref:
            if unit_ref == "|||":
                count_ref =count_ref +1
                ref_pos.append(pos_ref)
            else:
                pos_ref= pos_ref+1
        pos_in=0
        count=0
        for unit_in in word_in:
            if unit_in =="|||":
                count_in = count_in +1
                if pos_in in ref_pos:
                    count=count+1
                else:
                    continue
            else:
                pos_in = pos_in +1

        sum_count=sum_count+count
        sum_count_ref=sum_count_ref + count_ref
        sum_count_in = sum_count_in + count_in

        i=i+1

    P=sum_count/sum_count_in
    R=sum_count/sum_count_ref
    if (P+R) == 0:
        F1=0
    else:
        F1=(2*P*R)/(P+R)

    return P,R,F1
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file',type=str)
    parser.add_argument('out_file',type=str)
    parser.add_argument('ref_file',type=str)
    args = parser.parse_args()

    if os.path.isfile(args.in_file):
        fin = open(args.in_file,'rU')
    else:
        print("file [{}] not exist".format(args.in_file))
    
    if os.path.isfile(args.out_file):
        fout = open(args.out_file,'w')
    else:
        print("file [{}] not exist".format(args.out_file))
            
    string = ""
    for line in fin:
        word=line.strip().split()
        if len(word)==0:
            fout.write(string.strip()+'\n')
            string=""
        else:
            if word[2]=='yes':
                string = string +' '+word[0] + ' |||'
            else:
                string = string +' '+ word[0]
    fout.close()
                        
    P,R,F1=eval_method(args.out_file,args.ref_file)
    print("the whole text F1：%s,%s,%s"%(P,R,F1))
