lstm_units = 100

model = Sequential()
model.add(Embedding)
model.add(LSTM(lstm_units))
model.add(Dense, activation='sigmoid')
