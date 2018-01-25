# -*- coding: utf-8 -*-

from __future__ import print_function
import websocket
from sys import argv
import os
import time
class HLT():
    def __init__(self):
        self.websocket = None

    def get_web_content(self):
        self.websocket = websocket.create_connection("ws://127.0.0.1:8100")

    def shutdown_web_content(self):
        self.websocket.close()

    def get_ws(self,sentence):
        sentence = sentence.strip()
        sentence = "#$#WS#$# " + sentence
        self.websocket.send(sentence)
        result = self.websocket.recv()
        if result:
            ss = result.split("\t")
            segment = ss[0]
        else:
            segment = "no result received"
        return segment

    def get_pos(self,sentence):
        sentence = sentence.strip()
        sentence = "#$#POS#$# " + sentence
        self.websocket.send(sentence)
        result = self.websocket.recv()
        if result:
            ss = result.split("\t")
            segment = ss[0].split(" ")
            pos = ss[2].split(" ")
            sss = []
            if len(segment) == len(pos):
                for i in range(len(segment)):
                    sss.append(segment[i] + "/" + pos[i])
                ws_pos = " ".join(sss)
        else:
            ws_pos = "no result received"
        return ws_pos

    def get_dep(self,sentence):
        sentence = sentence.strip()
        self.websocket.send(sentence)
        result = self.websocket.recv()
        if result:
            ss = result.split("\t")
            segment = ss[0].split(" ")
            map = {}
            map.setdefault(0, "root")
            for i in range(len(segment)):
                map.setdefault(i + 1, segment[i])
                #print(i,segment[i])
            dep = ss[4].split(" ")
            #print('map=',map)
            list = []
            for dependency in dep:
                sss = dependency.split("_")
                sss[0] = map.get(int(sss[0]))
                sss[1] = map.get(int(sss[1]))
                sss=sss[:-2]
                list.append("".join(sss))
            s = "   ".join(list)
        else:
            s = "no result received"
        return s

if __name__ =='__main__':
    script,datafile = argv
    ss =HLT()
    ss.get_web_content()
    for fname in os.listdir(datafile):
        print(fname)
        num = 0
        time1 = time.time()
        b = open(fname+'_dep','w')
        for sentence in open(os.path.join(datafile,fname)):
            num += 1 
            if len(sentence.split())<20:
                sent = ss.get_dep(sentence.strip())
                print(sent,file=b)
            if num % 1000 ==0:
                time2 = time.time()
                print('cost time= ',time2-time1)
                time1 = time2
        b.close()
