import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

epoch_size = 768
test_size = int(epoch_size * 0.1)

data = np.array(pd.read_csv("data/diabetes.csv"))
train, test = train_test_split(data, test_size=test_size, random_state=41812)

y_train = (train[:, -1])
y_test = (test[:, -1])

print(y_train.shape)
print(y_test.shape)

X_train = np.delete(train, -1, 1)
X_test = np.delete(test, -1, 1)

inputs = keras.Input(X_train.shape[1], name="pima")
x = keras.layers.Dense(32, activation="relu")(inputs)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(64, kernel_regularizer=l2(0.01),
                       bias_regularizer=l2(0.01),
                       activation="relu")(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(64, kernel_regularizer=l2(0.01),
                       activation="relu")(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(32, bias_regularizer=l2(0.01),
                       activation="relu")(x)

outputs = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(
     loss=keras.losses.BinaryCrossentropy(),
     optimizer=keras.optimizers.Adam(),
     metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=8,  epochs=1000,
                    validation_split=0)

test_scores = model.evaluate(X_test, y_test, verbose=2)
