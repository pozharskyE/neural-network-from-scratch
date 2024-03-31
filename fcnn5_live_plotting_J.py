import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from live_plot import live_plot as live_plot_class

# N = [7, 5, 3, 1]
class NN():
  def __init__(self, N):
    # N - array containing num of nodes in each layer (N[0]-input layer; :hidden layers:, N[-1]-output layer)
    self.N = N
    self.L = len(N) - 1  # number of layers (without input layer)

    self.J_history = []
    self.J_cv_history = []

    # W[l].shape = (n[l], n[l-1])
    # W[1]
    # W_t = [W[1], W[2], ... , W[-1]]
    self.W_t = [[]] + [np.random.randn(N[l], N[l-1]) * (np.sqrt(2/N[l-1])) for l in range(1, self.L+1)]
    self.b_t = [[]] + [np.zeros((N[l], 1)) for l in range(1, self.L+1)]

    self.ReLU = np.vectorize(self.ReLU_function)
    self.ReLU_derivative = np.vectorize(self.ReLU_derivative_function)
    self.sigmoid = np.vectorize(self.sigmoid_function)

    self.zero_one_by_threshhold = np.vectorize(self.zero_one_by_threshhold_function)


  # These 3 functions are vectorized in __init__
  def ReLU_function(self, el):
    if el >= 0:
      return el
    else:
      return 0
    
  def ReLU_derivative_function(self, el):
    if el >= 0:
      return 1
    else:
      return 0
  
  def sigmoid_function(self, el):
    return (1 / (1 + np.exp(-el)))
  

  # Cost function(J) for optimizing + a couple of metrics for checking the progress 
  def J(self, y_pred, y_actual): # cost function
    if y_pred.shape != y_actual.shape:
      raise ValueError('y_pred.shape != y_actual.shape')
    return (- np.sum((y_actual*np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred)), axis=1, keepdims=True) / y_actual.shape[1])[0][0]

  def MAE(self, y_pred, y_actual):
    if y_pred.shape != y_actual.shape:
      raise ValueError('y_pred.shape != y_actual.shape')

    return (np.sum((y_pred - y_actual).__abs__(), axis=1, keepdims=True) / y_actual.shape[1])
  
  def zero_one_by_threshhold_function(self, el, threshhold=0.5):
    if el >= threshhold:
      return 1
    else:
      return 0

  def accuracy(self, y_pred, y_actual):
    if y_pred.shape != y_actual.shape:
      raise ValueError('y_pred.shape != y_actual.shape')
    y_pred_01 = self.zero_one_by_threshhold(y_pred)
    m = y_actual.shape[1]
    count = y_pred_01[y_pred_01==y_actual].size
    percent = round(count / m * 100, 2)
    return f'{count}/{m} ({percent}%)'


  def forward(self, X:np.array, keep_prob=1, return_cache=False):
    # Creating A and Z tensors, containing A, Z matrices for each layer
    # [[]] and [X] - are for 0 layer (Z[0] does not exist so its just None and A[0] is just input matrix X of shape (n[0], m))
    Z_t = [None] + [[] for _ in range(1, self.L + 1)]
    A_t = [X] + [[] for _ in range(1, self.L + 1)]

    # hidden layers
    for l in range(1, self.L):
      Z_t[l] = np.matmul(self.W_t[l], A_t[l-1]) + self.b_t[l]
      A_t[l] = self.ReLU(Z_t[l])

      # dropout regularization
      if keep_prob < 1:
        drop = np.random.rand(A_t[l].shape[0], A_t[l].shape[1]) < keep_prob
        A_t[l] = np.multiply(A_t[l], drop)
        A_t[l] = A_t[l] / keep_prob


    # last layer
    Z_t[-1] = np.matmul(self.W_t[-1], A_t[-2]) + self.b_t[-1]
    A_t[-1] = self.sigmoid(Z_t[-1])


    if return_cache:
      return A_t[-1], Z_t, A_t
    else:
      return A_t[-1]
      

  def backward(self, X, Y, l_rate=0.01, epochs=10, minibatch_size=128, keep_prob=1, X_cv=np.array([]), Y_cv=np.array([]), live_plot_every=0):

    start_Y_pred = self.forward(X)
    J = self.J(start_Y_pred, Y)
    mae = self.MAE(start_Y_pred, Y)
    accuracy = self.accuracy(start_Y_pred, Y)
    self.J_history.append(J)

    cv_set_was_given = X_cv.any() and Y_cv.any()
    if cv_set_was_given:
      start_Y_pred_cv = self.forward(X_cv)
      J_cv = self.J(start_Y_pred_cv, Y_cv)
      mae_cv = self.MAE(start_Y_pred_cv, Y_cv)
      accuracy_cv = self.accuracy(start_Y_pred_cv, Y_cv)
      self.J_cv_history.append(J_cv)

    
    if live_plot_every:
      live_plot = live_plot_class()


    for epoch in range(1, epochs+1):
      print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
      print(f'*Epoch {epoch}/{epochs}*')

      # Check shapes
      if X.shape[0] != self.N[0] or Y.shape[0] != self.N[-1] or X.shape[1] != Y.shape[1]:
        raise ValueError('X should have shape (n[0], m) and Y (n[-1], m)')
      
      m = X.shape[1]

      num_minibatches = m // minibatch_size
      X_to_split = X[:, :minibatch_size * num_minibatches]
      Y_to_split = Y[:, :minibatch_size * num_minibatches]

      X_splitted = np.split(X_to_split, num_minibatches, axis=1)
      Y_splitted = np.split(Y_to_split, num_minibatches, axis=1)

      for t in range(num_minibatches):
        y_pred, Z_t, A_t = self.forward(X_splitted[t], keep_prob=keep_prob, return_cache=True)


        dA_t = [[] for _ in range(0, self.L)] + [None]  # dont need dA of the last layer (skip to dZ)
        dZ_last = A_t[-1] - Y_splitted[t]
        dZ_t = [None] + [[] for _ in range(1, self.L)] + [dZ_last]

        dW_last = np.matmul(dZ_last, A_t[-2].T) / minibatch_size
        db_last = np.sum(dZ_last, axis=1, keepdims=True) / minibatch_size
        self.W_t[-1] -= l_rate * dW_last
        self.b_t[-1] -= l_rate * db_last

        dW_t = [None] + [[] for l in range(1, self.L)] + [dW_last]
        db_t = [None] + [[] for l in range(1, self.L)] + [db_last]


        for l in range(self.L - 1, 0, -1):
          dA_t[l] = np.matmul(self.W_t[l+1].T, dZ_t[l+1])

          dZ_t[l] = dA_t[l] * self.ReLU_derivative(Z_t[l])

          dW_t[l] = np.matmul(dZ_t[l], A_t[l-1].T) / minibatch_size
          db_t[l] = np.sum(dZ_t[l], axis=1, keepdims=True) / minibatch_size


          self.W_t[l] -= l_rate * dW_t[l]
          self.b_t[l] -= l_rate * db_t[l]
      
      
        
        # Check a progress 
        new_y_pred = self.forward(X)
        new_J = self.J(new_y_pred, Y)
        new_mae = self.MAE(new_y_pred, Y)
        new_accuracy = self.accuracy(new_y_pred, Y)

        self.J_history.append(new_J)

        J = new_J
        mae = new_mae
        accuracy = new_accuracy


        if cv_set_was_given:
          new_y_pred_cv = self.forward(X_cv)
          new_J_cv = self.J(new_y_pred_cv, Y_cv)
          new_mae_cv = self.MAE(new_y_pred_cv, Y_cv)
          new_accuracy_cv = self.accuracy(new_y_pred_cv, Y_cv)


          self.J_cv_history.append(new_J_cv)

          J_cv = new_J_cv
          mae_cv = new_mae_cv
          accuracy_cv = new_accuracy_cv


        # Printing a progress (united because there was a delay that causes shifting in printing the progress during training (because of computation))
        if cv_set_was_given:
          print('-------------------------------------------')
          print(f'[ep{epoch}/{epochs}] Minibatch {t+1}/{num_minibatches}')

          print(f'J: {J} => {new_J}')
          print(f'MAE: {mae} => {new_mae}')
          print(f'Accuracy: {accuracy} => {new_accuracy}')

          print(f'J_cv: {J_cv} => {new_J_cv}')
          print(f'MAE_cv: {mae_cv} => {new_mae_cv}')
          print(f'Accuracy_cv: {accuracy_cv} => {new_accuracy_cv}')

        else:
          print('-------------------------------------------')
          print(f'[ep{epoch}/{epochs}] Minibatch {t+1}/{num_minibatches}')

          print(f'J: {J} => {new_J}')
          print(f'MAE: {mae} => {new_mae}')
          print(f'Accuracy: {accuracy} => {new_accuracy}')

        if live_plot_every and (t % live_plot_every == 0):
          live_plot.update(self.J_history, self.J_cv_history)

          
    self.J_history.append(new_J)

    if cv_set_was_given:
      self.J_cv_history.append(new_J_cv)
      live_plot.update(self.J_history, self.J_cv_history)

      


  def save_json(self, file_path='./model_params.json'):
    params = {'W_t': [[]] + [np_arr.tolist() for np_arr in self.W_t[1:]], 'b_t': [[]] + [np_arr.tolist() for np_arr in self.b_t[1:]], 'N': self.N, 'L': self.L}
    with open(file_path, 'w') as file:
      json.dump(params, file)

    
  def load_json(self, file_path='./model_params.json', with_N_and_L=True):
    with open(file_path, 'r') as file:
      params = json.load(file)
      self.W_t = [[]] + [np.array(lst) for lst in params['W_t'][1:]]
      self.b_t = [[]] + [np.array(lst) for lst in params['b_t'][1:]]

      if with_N_and_L:
        self.N = params['N']
        self.L = params['L']
        