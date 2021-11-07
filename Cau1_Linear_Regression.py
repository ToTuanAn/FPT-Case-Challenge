from keras.models import Sequential
from keras.layers import Dense

def build_linear_regression_model(input_dim):
  input_dim = tuple((input_dim,))
  model = Sequential([Dense(1, input_shape=input_dim, activation=None, bias_initializer=tf.keras.initializers.GlorotNormal())])
  return model
