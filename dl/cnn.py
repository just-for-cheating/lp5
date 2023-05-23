import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

df_train = pd.read_csv("fashion-mnist_train.csv")
df_test = pd.read_csv("fashion-mnist_test.csv")

# Visualize random products
training = np.asarray(df_train.drop('label', axis=1))
train_label = np.asarray(df_train['label'])

testing = np.asarray(df_test.drop('label', axis=1))
test_label = np.asarray(df_test['label'])

df_train.head()

set(train_label)

nrows = 10
ncols = 10
n = len(training)
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))

for i in range(nrows):
    for j in range(ncols):
        index = random.randint(0, n)
        axes[i][j].imshow(training[index].reshape(28, 28))
        axes[i][j].set_title(train_label[index])
        axes[i][j].axis("off")
plt.subplots_adjust(hspace=0.5)

training = training/255

testing = testing/255

training.shape

training = np.reshape(training, (-1, 28, 28, 1))
testing = np.reshape(testing, (-1, 28, 28, 1)) 

testing.shape

# Split the data into train and val data
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(training, train_label, test_size=0.2, random_state=1)

x_train.shape

# Model creation
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))

model.evaluate(testing, test_label)

y_pred = np.argmax(model.predict(testing), axis=-1)

nrows = 10
ncols = 10

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))

#Visualize the prediction
for i in range(nrows):
    for j in range(ncols):
        index = random.randint(0, len(testing))
        axes[i][j].imshow(testing[index].reshape(28, 28))
        axes[i][j].set_title("Actual : {:d}\n Predicted : {:d}".format(test_label[index], y_pred[index]))
        axes[i][j].axis("off")
plt.subplots_adjust(hspace=0.5)

from sklearn.metrics import classification_report
print(classification_report(test_label, y_pred))

