# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("HousingData.csv")
df.head()

# Info. about dataset
df.info()

# Describe dataset
df.describe()

# Data Cleaning
df.isnull().sum()

df.head()

crim_avg = round(df['CRIM'].mean(),5)
crim_avg

df['CRIM'] = df['CRIM'].fillna(crim_avg)
df['CRIM'].isnull().sum()

# Function to fill all nan values
def auto_fillna(col, df):
    col_avg = round(df[col].mean(), 5)
    return df[col].fillna(col_avg)

cols = ['ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT']
for ele in cols:
    df[ele] = auto_fillna(ele, df)
print("Data cleaned!")

df.isnull().sum()

# Data Visualization
plt.hist(df['MEDV'])
plt.show()

plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True)
plt.show()

sns.boxplot(data=df['MEDV'])
plt.show()

# Split the data into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1), df['MEDV'], test_size=0.2, random_state=1)

# Data Strandalization
from sklearn.preprocessing import StandardScaler
scScale = StandardScaler()
x_train = scScale.fit_transform(x_train)
x_test = scScale.transform(x_test)

# from sklearn.preprocessing import MinMaxScaler
# sc=StandardScaler()
# sc1 = MinMaxScaler()
# y_train = sc1.fit_transform(y_train.values.reshape(-1,1))
# y_test = sc1.transform(y_test.values.reshape(-1,1))

# Model creation
from keras.models import Sequential
# Add layers
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=240, activation='relu', input_shape=(13,)))
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=60, activation='relu'))
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=300, batch_size=32)

model.evaluate(x_test, y_test)