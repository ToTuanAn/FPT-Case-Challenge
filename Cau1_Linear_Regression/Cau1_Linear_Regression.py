from keras.models import Sequential
from keras.layers import Dense

#phương thức để xây dựng model linear regression
def build_linear_regression_model(input_dim):
  input_dim = tuple((input_dim,))
  model = Sequential([Dense(1, input_shape=input_dim, activation=None, bias_initializer=tf.keras.initializers.GlorotNormal())])
  return model

#thư viện cần cài đặt
from sklearn.datasets import load_diabetes
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#normalized dữ liệu
def normalise(data):
  m = np.mean(data)
  n = np.std(data)
  norm = (data -m)/ n
  return norm

#lấy dữ liệu từ diabetes.csv từ sklearn
def get_data():
  diabetes = load_diabetes()
  features = np.array(diabetes.data)
  labels = np.array(diabetes.target)
  labels = np.reshape(labels, (-1,1))
  labels = normalise(labels)
  return features,labels

X,Y = get_data()

#xây dựng model
model = build_linear_regression_model(X.shape[1])

#chạy model
model.compile(optimizer='adam', loss='mse')
hst = model.fit(X,Y,batch_size = 8, epochs = 200, shuffle=True, validation_split=0.1)

plt.plot(hst.history['loss'], 'g')
plt.plot(hst.history['val_loss'], 'r')

#kết quả r2 của model
print("Kết quả thuật toán: ")
print(r2_score(y_pred=model.predict(X), y_true=Y))