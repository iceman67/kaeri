import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def concat_dataframes(df1, df2, axis=0, join="outer", ignore_index=False):
    """
    Concatenates two pandas DataFrames.

    Args:
        df1 (pandas.DataFrame): The first DataFrame.
        df2 (pandas.DataFrame): The second DataFrame.
        axis (int, optional): The axis to concatenate along (0 for rows, 1 for columns). Defaults to 0.
        join (str, optional): How to handle indexes on other axis(es) (outer or inner). Defaults to 'outer'.
        ignore_index (bool, optional): If True, do not use the index values along the concatenation axis. The resulting axis will be labeled 0, ..., n - 1. Defaults to False.

    Returns:
        pandas.DataFrame: The concatenated DataFrame.
    """
    return pd.concat([df1, df2], axis=axis, join=join, ignore_index=ignore_index)


data_dir = "./"
data = np.load(
    data_dir + "normal_data.npy", allow_pickle=True
)  # 'your_data.npy' 파일 경로를 적절히 수정
df_normal = pd.DataFrame(data)
df_normal[320] = 1

data = np.load(
    data_dir + "leak_data.npy", allow_pickle=True
)  # 'your_data.npy' 파일 경로를 적절히 수정
df_leak = pd.DataFrame(data)
df_leak[320] = 0


data = concat_dataframes(df_normal, df_leak)
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

np.savez("kaeri.npz", data.to_numpy())


model = tf.keras.models.Sequential(
    [
        # 한 개의 층으로만 구성
        tf.keras.layers.Dense(1, input_dim=320, activation="sigmoid"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)
sc_X = StandardScaler()
X_trainscaled = sc_X.fit_transform(X_train)
X_testscaled = sc_X.transform(X_test)

hist = model.fit(X_trainscaled, y_train, epochs=200)

print(f"accuracy = {max(hist.history['accuracy'])}")

print(model.summary())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history["loss"])
plt.title("loss")
plt.subplot(1, 2, 2)
plt.title("accuracy")
plt.plot(hist.history["accuracy"], "b-", label="training")
plt.legend()
plt.show()
