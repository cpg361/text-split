import os,sys
sys.path.append("..")
import gensim
import numpy as np
from singleton import Singleton

class BenebotVector(metaclass=Singleton):

    model = None

    def __init__(self, model_path):
        if not self.model:
            print('load word vector model')
            self.model = gensim.models.Word2Vec.load(model_path)
        #self.dim = len(self.getVectorByWord('中国'))
        self.dim = self.model.vector_size

    def hasWord(self, word):
        result = self.model.__contains__(word.strip())
        if result:
            return 1
        return 0

    def getUnkWords(self, words):
        result = []
        for word in words:
            if not self.hasWord(word):
                result.append(word)
        return result

    def getVectorByWord(self, word):
        result = []
        if self.model.__contains__(word.strip()):
            vector = self.model.__getitem__(word.strip())
            result = [float(v) for v in vector]
        return result

    def getVectorByWords(self, words):
        result = {}
        for word in words:
            result[word] = self.getVectorByWord(word)
        return result

    def getSimilarWords(self, word):
        result = []
        if self.model.__contains__(word.strip()):
            result = self.model.most_similar(word.strip())
        return result

    def calWordSimilarity(self, words):
        l = words.split('|')
        result = 0
        if len(l) > 1:
            result = self.model.similarity(l[0].strip(), l[1].strip())
        return result

    def getVectorBySentence(self, sentence):
        words = sentence.strip().split(' ')
        vectors = []
        for word in words:
            vector = self.getVectorByWord(word)
            if not vector:
                vector = self.getVectorByWord('unk')
            if not vector:
                vector = [0.0] * self.dim
            vectors.append(vector)
        result = [0.0] * self.dim
        if vectors:
            result = np.mean(vectors, axis = 0).tolist()
        return result

    def getVectorByWeightSentence(self, weight_sentence):
        vectors = []
        value_sum = 0.0
        for key,value in weight_sentence.items():
            vector = self.getVectorByWord(key)
            if not vector:
                vector = self.getVectorByWord('unk')
            if not vector:
                vector = np.array([0.0] * self.dim)
            vector = np.array(vector) * value
            vectors.append(vector)
            value_sum += value
        result = np.array([0.0] * self.dim)
        if vectors:
            result = np.sum(vectors, axis = 0) / value_sum
        result = result.tolist()
        return result

if __name__ == '__main__':
    bv = BenebotVector('word2vec.bin')
    result = bv.getVectorBySentence('我 是 中国 人')
    print(result)
