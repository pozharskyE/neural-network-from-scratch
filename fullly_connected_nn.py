import numpy as np
# N = [7, 5, 3, 1]
class NN():
  def __init__(self, N, W_init_scale=0.1):
    # N - array containing num of nodes in each layer (N[0]-input layer; :hidden layers:, N[-1]-output layer)
    self.N = N
    self.L = len(N) - 1  # number of layers (without input layer)

    # W[l].shape = (n[l], n[l-1])
    # W[1]
    # W_t = [W[1], W[2], ... , W[-1]]
    self.W_t = [[]] + [np.random.randn(N[l], N[l-1]) * W_init_scale for l in range(1, self.L+1)]
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
    # print(f'J = {- np.sum((y_actual*np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred)), axis=1, keepdims=True)} / {y_actual.shape[1]}')
    # print(f'np.log(y_pred) = {np.log(y_pred)}')
    return (- np.sum((y_actual*np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred)), axis=1, keepdims=True) / y_actual.shape[1])

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


  def forward(self, X:np.array, return_cache=False):
    # Creating A and Z tensors, containing A, Z matrices for each layer
    # [[]] and [X] - are for 0 layer (Z[0] does not exist so its just None and A[0] is just input matrix X of shape (n[0], m))
    Z_t = [None] + [[] for _ in range(1, self.L + 1)]
    A_t = [X] + [[] for _ in range(1, self.L + 1)]

    for l in range(1, self.L):
      Z_t[l] = np.matmul(self.W_t[l], A_t[l-1]) + self.b_t[l]
      # print(f'Z_t[{l}] now = {Z_t[l]}')
      A_t[l] = self.ReLU(Z_t[l])
      # print(f'A_t[{l}] now = {A_t[l]}')

    Z_t[-1] = np.matmul(self.W_t[-1], A_t[-2]) + self.b_t[-1]
    A_t[-1] = self.sigmoid(Z_t[-1])


    if return_cache:
      return A_t[-1], Z_t, A_t
    else:
      return A_t[-1]
      

  def backward(self, X, Y, l_rate=0.01, epochs=10):
    J_history = []
    for epoch in range(1, epochs+1):
      # print(f'*Epoch {epoch} begins*')
      # Check shapes
      if X.shape[0] != self.N[0] or Y.shape[0] != self.N[-1] or X.shape[1] != Y.shape[1]:
        raise ValueError('X should have shape (n[0], m) and Y (n[-1], m)')
      
      m = X.shape[1]

      y_pred, Z_t, A_t = self.forward(X, return_cache=True)

      # Save old values of W, b, J for printing results of this epoch
      old_J = self.J(y_pred, Y)
      J_history.append(old_J)
      old_MAE = self.MAE(y_pred, Y)
      old_accuracy = self.accuracy(y_pred, Y)


      dA_t = [[] for _ in range(0, self.L)] + [None]  # dont need dA of the last layer (skip to dZ)
      dZ_last = A_t[-1] - Y
      dZ_t = [None] + [[] for _ in range(1, self.L)] + [dZ_last]

      dW_last = np.matmul(dZ_last, A_t[-2].T) / m
      db_last = np.sum(dZ_last, axis=1, keepdims=True) / m
      self.W_t[-1] -= l_rate * dW_last
      self.b_t[-1] -= l_rate * db_last

      dW_t = [None] + [[] for l in range(1, self.L)] + [dW_last]
      db_t = [None] + [[] for l in range(1, self.L)] + [db_last]


      for l in range(self.L - 1, 0, -1):
        dA_t[l] = np.matmul(self.W_t[l+1].T, dZ_t[l+1])

        dZ_t[l] = dA_t[l] * self.ReLU_derivative(Z_t[l])

        dW_t[l] = np.matmul(dZ_t[l], A_t[l-1].T) / m
        db_t[l] = np.sum(dZ_t[l], axis=1, keepdims=True) / m


        self.W_t[l] -= l_rate * dW_t[l]
        self.b_t[l] -= l_rate * db_t[l]
      
      # print changes of W, b, J and MAE
      new_y_pred = self.forward(X)
      new_J = self.J(new_y_pred, Y)
      new_MAE = self.MAE(new_y_pred, Y)
      new_accuracy = self.accuracy(new_y_pred, Y)

      print('----------------------------------')
      print(f'Changes in epoch {epoch}:')
      print(f'J: {old_J} => {new_J}')
      print(f'MAE: {old_MAE} => {new_MAE}')
      print(f'Accuracy: {old_accuracy} => {new_accuracy}')

    J_history.append(new_J)
    return J_history

# if __name__ == '__main__':
#   tar = np.array([[45, 53.3333333333, -100, 3, 0, 66, -495], [54, 5, -10, 33, 0, 66, -4]]).T
#   # model_1.J(tar, np.array([[1, 1, 3], [43, 6, -99]]))
#   model_1 = NN([7, 5, 3, 1])
#   print(model_1.forward(tar))