import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle
from keras.models import load_model
pd.set_option('display.unicode.east_asian_width', True) # 이걸주면 어느정도 줄 맞춰줌 추가된
pd.set_option('display.max_columns', 15)


df = pd.read_csv('./crawling_data/naver_headline_news_20221128.csv') # 오늘자 헤드라인 뉴스 가져오기
print(df.head())
df.info()

X = df['titles']
Y = df['category']

with open('./models/labled_encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)
labeled_Y = encoder.transform(Y)
onehot_Y = to_categorical(labeled_Y)
# 원화된 코딩할 때 전처리할 때 썼던 것 그대로 써야함. models>라벨
# 이제 핏을 하면 안됨

okt = Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)

with open('./models/news_token_pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 20:
        tokened_X[i] = tokened_X[i][:20]
X_pad = pad_sequences(tokened_X, 20)

model = load_model('./models/news_category_classfication_model_0.993.h5') #나는 왜 993? 달라도됨
preds = model.predict(X_pad)
label = encoder.classes_
category_preds = []
for pred in preds:
    category_pred = label[np.argmax(pred)]
    category_preds.append(category_pred)
df['predict'] = category_preds




df['OX'] = False
for i in range(len(df)):
    if df.loc[i, 'category'] == df.loc[i, 'predict']:   #원래 카테고리값이랑 예측값이랑 같으면 트루
        df.loc[i, 'OX'] = True

print(df.head(30))
print(df['OX'].value_counts()) #오엑스 컬럼에 ..를 해보면 몇개 맞고 틀린 지 알 수 있
print(df['OX'].mean()) #정답률
print(df.loc[df['OX']==False])  # 틀린것만 출력되도록!




# 길이제한 20개
# 만약 20보다크면 토큰드x의 i를 20개까지만 슬라이싱



# 원화된코드를 위해... 라벨을 가지고 있는 라벨인코더
# ..가 카테고리값이 되는 것
# 카테고리와 00만 있는데, 컬럼과 섹션설명?도









