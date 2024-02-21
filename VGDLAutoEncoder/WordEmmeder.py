
from numpy import array

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from DescriptionToTokens import get_codes
import keras 
# define documents
codes, token_decoder = get_codes("examples/all_games_sp.csv" )
lengths = [len(code) for code in codes]
max_len = max(lengths) 
padded_docs = pad_sequences(codes, maxlen=max_len, padding='post')
 

# integer encode the documents
vocab_size = token_decoder.next_token
encoded_docs = keras.utils.to_categorical(padded_docs, num_classes=vocab_size)
print(encoded_docs[0])
# pad documents to a max length of 4 words

# define the model
model = Sequential()
model.add(keras.layers.Input(vocab_size))
model.add(Embedding(vocab_size, 30, input_length=max_len))
model.add(Flatten())
model.add(Dense(max_len, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(encoded_docs, encoded_docs, epochs=200, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, encoded_docs, verbose=0)
print('Accuracy: %f' % (accuracy*100))