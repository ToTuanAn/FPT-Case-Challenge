# Thư viện cần cài đặt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Chi tiết về dữ liệu mẫu https://archive.ics.uci.edu/ml/datasets/diabetes
# Sửa lại tên file và tham số sep dựa vào ký tự ngăn cách các từ, nếu dữ liệu không có tên cột thì thêm tham số header = None
def get_data():
    data = pd.read_csv('input.txt' , sep = ',')
    features = data.loc[: , data.columns.drop(['id' , 'target'])].values
    labels = data.loc[: , 'target'].values
    labels = np.array(diabetes.target)
    labels = np.reshape(labels, (-1,1))
    labels = normalise(labels)
    return features,labels

# Normalized dữ liệu
def normalise(data):
    m = np.mean(data)
    n = np.std(data)
    norm = (data -m)/ n
    return norm

# Phương thức để xây dựng model linear regression
def build_linear_regression_model(input_dim):
    input_dim = tuple((input_dim,))
    model = Sequential([Dense(1, input_shape=input_dim, activation=None, bias_initializer=tf.keras.initializers.GlorotNormal())])
    return model

X,Y = get_data()

#xây dựng model
model = build_linear_regression_model(X.shape[1])

#chạy model
model.compile(optimizer='adam', loss='mse')
hst = model.fit(X,Y,batch_size = 8, epochs = 200, shuffle=True, validation_split=0.1)

plt.plot(hst.history['loss'], 'g' , label = 'train loss')
plt.plot(hst.history['val_loss'], 'r' , label = 'validation loss')
plt.legend()
plt.show()

#kết quả r2 của model
print("Kết quả thuật toán: ")
print(r2_score(y_pred=model.predict(X), y_true=Y))
