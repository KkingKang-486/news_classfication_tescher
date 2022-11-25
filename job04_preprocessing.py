import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle

pd.set_option('display.unicode.east_asian_width', True) #줄맞추기
df = pd.read_csv('./crawling_data/naver_news_titles_20221125.csv') #디에프 만들어서 봅시다
print(df.head())
print(df.category.value_counts())
df.info()

X = df['titles']
Y = df['category']

encoder = LabelEncoder()  #아직 컨캣은 안한상태 실험해보기 어렵 11:10 # 데이터 전처리, 와이가 간단하니 먼저
labeled_Y = encoder.fit_transform(Y) # 핏트랜스폼하면 정보를 가지게 됨
print(labeled_Y[:5])
print(encoder.classes_)
with open('./models/labled_encoder.pickle', 'wb') as f: #원화된 코딩하려고
    pickle.dump(encoder, f)
onehot_Y = to_categorical(labeled_Y)
print(onehot_Y[:5])

okt = Okt()

for i in range(len(X[5:])):
    X[i] = okt.morphs(X[i], stem=True)      # X[i]로 덮어쓰기
    if i % 100 == 0:
        print('.', end='')
    if i % 10000 == 0:              #1000=>10000 (10줄나오면 끝날것)
        print()

                                    #점 10개찍고 줄바꾸고, 인덱싱없애기?

 #print(X[:10]) #10개만 봐보기
    # X[i] = okt.morphs(X[1111], stem=True) #X[i]로 덮어쓰기
    # print(X[1111])
    # print(okt_morph_X)

stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)):
    words =[]
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:        #j i 순서로 가도록, 형태소길이가 1개 이상이면 words에 넣어줌
            if X[j][i] not in stopwords['stopword']:    # 불용어 사전에 있지 많으면
                words.append(X[j][i])
    X[j] = ' '.join(words)

token = Tokenizer()     #()안하면 토큰이 객체가 아님******
token.fit_on_texts(X)   # 뭔가 바꿔주는 애들은 핏트랜스폼해야. 엑스주면 엑스 안에 있는 형태소들 다 끄집어냄 = 집합 백오브워즈 BOW
tokened_X = token.texts_to_sequences(X)  # 순서가 있어야 하니
wordsize = len(token.word_index) + 1
with open('./models/news_token_pickle', 'wb') as f:
    pickle.dump(token, f)       #토큰저장
# print(tokened_X)
# print(wordsize)

max_len = 0
for i in range(len(tokened_X)):             #토큰드 엑스의 길이만큼 돌면서 제일긴문장의 길이 알아내야
    if max_len < len(tokened_X[i]):
        max_len = len(tokened_X[i])
print(max_len)


X_pad = pad_sequences(tokened_X, max_len)   #앞에 영으로 패딩을 입혀서 제일 긴문장의 길이만큼 맞춰주는
# print(X_pad)


X_train, X_test, Y_train, Y_test = train_test_split(
    X_pad, onehot_Y, test_size=0.1)
print(X_train.shape, Y_train.shape, X_train.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./models/news_data_max_{}_wordsize_{}.npy'.format(max_len, wordsize), xy)
#모델스폴더에 저장 # 뭘 저장할 지 안정했던 = xy












