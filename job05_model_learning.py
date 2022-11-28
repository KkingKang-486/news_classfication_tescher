import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './models/news_data_max_20_wordsize_11919.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(11919, 300, input_length=20)) # 본인단어 개수로 난 11919
# 임베딩 레이어의 역할 : 수치적으로 계산하여안되는 명목척도
# 의미공간상의 배치, 11919차원=> 300차원으로 줄일 것. => 각각의 형태소가 의미를 계산할 수 있게됨
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))       # 1써주면 달라지는 거 없지만 그래도 습관적으로
model.add(GRU(128, activation='tanh', return_sequences=True))       #리턴시퀀스는 나중에 설명
model.add(Dropout(0.3))      # 과접합 막기위해
model.add(GRU(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(64, activation='tanh'))   #GRU는 여기까지
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu')) # 좀 딥러닝같다
model.add(Dense(6, activation='softmax'))     # 카테고리 여섯개
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128,
                     epochs=10, validation_data=(X_test, Y_test))
model.save('./models/news_category_classfication_model_{}.h5'.format(
    np.round(fit_hist.history['val_accuracy'][-1], 3))) #한참돌렸는데 저장안하면 말짱꽝, 중간에 팅겨도!
#소수점아래 3째까지
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()











