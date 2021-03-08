import numpy as np
import time
import codecs
import jieba
import re
import csv


# preprocessing(segmentation，stopwprds filtering)
def preprocessing():
    # read the list of stopwords
    file = codecs.open('./datasets/stopwords.dic', 'r', 'utf-8')
    stopwords = [line.strip() for line in file]
    file.close()

    # read the corpus for training
    file = codecs.open('./datasets/dataset_cn.txt', 'r', 'utf-8')
    documents = [document.strip() for document in file]
    file.close()

    word2id = {}
    id2word = {}
    docs = []
    currentDocument = []
    currentWordId = 0

    for document in documents:
        # segmentation
        segList = jieba.cut(document)
        for word in segList:
            word = word.lower().strip()
            # filter the stopwords
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
                if word in word2id:
                    currentDocument.append(word2id[word])
                else:
                    currentDocument.append(currentWordId)
                    word2id[word] = currentWordId
                    id2word[currentWordId] = word
                    currentWordId += 1
        docs.append(currentDocument)
        currentDocument = []
    return docs, word2id, id2word


# initialization, sample from uniform multinomial
def randomInitialize():
    for d, doc in enumerate(docs):
        zCurrentDoc = []
        for w in doc:
            pz = np.divide(np.multiply(ndz[d, :], nzw[:, w]), nz)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            zCurrentDoc.append(z)
            ndz[d, z] += 1
            nzw[z, w] += 1
            nz[z] += 1
        Z.append(zCurrentDoc)


# gibbs sampling
def gibbsSampling():
    # re-sample every word in every document
    for d, doc in enumerate(docs):
        for index, w in enumerate(doc):
            z = Z[d][index]
            ndz[d, z] -= 1
            nzw[z, w] -= 1
            nz[z] -= 1

            pz = np.divide(np.multiply(ndz[d, :], nzw[:, w]), nz)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            Z[d][index] = z

            ndz[d, z] += 1
            nzw[z, w] += 1
            nz[z] += 1


def perplexity():
    nd = np.sum(ndz, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(docs):
        for w in doc:
            ll = ll + np.log(((nzw[:, w] / nz) * (ndz[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll / (-n))


alpha = 5
beta = 0.1
iterationNum = 20
Z = []
K = 10
docs, word2id, id2word = preprocessing()
N = len(docs)
M = len(word2id)
ndz = np.zeros([N, K]) + alpha
nzw = np.zeros([K, M]) + beta
nz = np.zeros([K]) + M * beta
randomInitialize()
for i in range(0, iterationNum):
    gibbsSampling()
    print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", perplexity())

topicwords = []
maxTopicWordsNum = 10
for z in range(0, K):
    ids = nzw[z, :].argsort()
    topicword = []
    for j in ids:
        topicword.insert(0, id2word[j])
    topicwords.append(topicword[0: min(10, len(topicword))])

with open('./datasets/topicword.csv', 'w', encoding='utf-8', newline='') as f:
    csv_write = csv.writer(f)
    for topicword in topicwords:
        csv_write.writerow(topicword)
