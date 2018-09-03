import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


data = pd.read_csv('data.txt',sep = ',',names = ['category','tweet'])  # insert all the data into a data frame


data = data[data.category != "Both"]                             # remove the tweets that belong to Both category
data['tweet'] = data['tweet'].apply(lambda x: x.lower())         # convert the tweets in lower case
data['tweet'] = data['tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x))) # remove punctuation and keep strings and digits

print(data[data['category'] == 'Neither'].size)
print(data[data['category'] == 'Racism'].size)
print(data[data['category'] == 'Sexism'].size)

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')


max_fatures = 20000                                             # the most common unique words we will use for our model
tokenizer = Tokenizer(num_words = max_fatures,lower=True, split=' ') # split the tweets into words and take only the most 20000 common words 
tokenizer.fit_on_texts(data['tweet'].values)
print(tokenizer.word_index)                                    # each word appear with an integer next to it
X = tokenizer.texts_to_sequences(data['tweet'].values)  # convert the text to sequences
X = pad_sequences(X)                                   # convert the sequences into 2-D numpy array
print(X[0])


#create the RNN model - features
embed_dim = 128
lstm_out = 196

print (X.shape[1])

model = Sequential()                                   # linear stack of layers - topology of the network
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1])) #first layer - creating words vectors
model.add(SpatialDropout1D(0.4))                       #prevent overfitting
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))  #build the rnn model
model.add(Dense(3,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


Y = pd.get_dummies(data['category']).values    # get the category of each tweet
print(Y[0])
print(Y[2])
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 32
model.fit(X_train, Y_train, epochs = 15, batch_size=batch_size, verbose = 2) # train the model



validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)  #evaluate the model on the test data to find the accuracy
print("score: %.2f" % (score))
print("acc: {} ".format (acc*100))


twt = ["all psychopaths have a female brain"]

twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=104, dtype='int32', value=0)   #padding the tweet to have exactly the same shape as `embedding_1` input


sentiment = model.predict(np.array(twt))
print(sentiment)
B = np.where(sentiment > 0.5, 1, 0)

print(B)
print(type(B))
print(type(Y[0]))

if(B.item(0)==1):
	print("Neither")
elif(B.item(1)==1):
	print("Racism")
else:
    print("Sexism")



