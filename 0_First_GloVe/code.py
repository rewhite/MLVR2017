import os

def loadGloveModel():
    print("Loading Model")
    f = open('glove.6B.50d.txt')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done")
    return model

# model = loadGloveModel()
# print(model['cat'])
git
squad 데이터셋 로드
한 지문을 NLTK tokenizer 사용해서 토큰으로 자름
각 토큰에 해당하는 워드 벡터 로딩
