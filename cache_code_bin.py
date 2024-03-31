def MSE(y_pred, y_actual):
  if y_pred.shape != y_actual.shape:
    raise ValueError('y_pred.shape != y_actual.shape')
  
  return (np.sum((y_pred - y_actual)**2, axis=1, keepdims=True) / y_actual.shape[1])

print(MSE(model_1_out, y_train))

def z_standartization(df, columns):
  mean_dict = {}
  sd_dict = {}
  new_df = df.copy()
  for col in columns:
    c = np.array(new_df[col])
    mean = c.mean()
    print(mean, c.mean())
    sd = np.sqrt(sum((c - mean)**2) / len(c))
    new_df[col] = (c - mean) / sd

    mean_dict[col] = mean
    sd_dict[col] = sd
  return new_df, mean_dict, sd_dict

# ['CreditScore', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
scaled_X_train, mean_X_train, sd_X_train = z_standartization(X_train, ['CreditScore', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'])

def correlation_r(x, y):
  m = len(x)
  mean_x = sum(x) / m
  mean_y = sum(y) / m
  sd_x = np.sqrt(sum((x - mean_x)**2) / m) # standart deviation
  sd_y = np.sqrt(sum((y - mean_y)**2) / m)
  z_score_x = (x - mean_x) / sd_x
  z_score_y = (y - mean_y) / sd_y
  r = sum(z_score_x * z_score_y) / m
  return r

def matrix_norm_2(matrix:np.array):  # aka Frobenius norm squared
  return (matrix**2).sum()


y_pred_cv = model_2_L2.forward(scaled_X_cv)

def precision_recall(y_pred, y_actual, threshhold=0.8):
  y_pred_01 = model_2_L2.zero_one_by_threshhold(y_pred, threshhold=threshhold)

  tp = np.count_nonzero(np.logical_and(y_pred_01==1, y_actual==1))
  fp = np.count_nonzero(np.logical_and(y_pred_01==1, y_actual==0))
  fn = np.count_nonzero(np.logical_and(y_pred_01==0, y_actual==1))

  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  return precision, recall

precision, recall = precision_recall(y_pred_cv, y_cv, threshhold=0.8)
precision, recall

def save_json(model):
  params = {'W_t': [[]] + [np_arr.tolist() for np_arr in model.W_t[1:]], 'b_t': [[]] + [np_arr.tolist() for np_arr in model.b_t[1:]], 'N': model.N, 'L': model.L}
  with open('./model_params.json', 'w') as file:
    json.dump(params, file)

save_json(model_2_L2)

'''
#### model_1 = NN([7, 5, 3, 1]), epochs=1000, l_rate=0.1
result MAE: [[0.33266521]]
#### model_1 = NN([7, 10, 3, 1]), epochs=1000, l_rate=0.1
the same as previous
#### model_1 = NN([7, 10, 10, 3, 1]), l_rate=0.5, after 200 epochs
result MAE: [[0.33266521]]
#### model_1 = NN([7, 30, 50, 10, 1]), l_rate=0.5, after 100 epochs
[[0.33268058]]
#### model_1 = NN([7, 10, 30, 50, 30, 10, 1]), l_rate=1, after 130 epochs
[[0.33266521]]
#### model_1 = NN([7, 5, 3, 1]), l_rate=1, after 187 epochs
[[0.33266509]]
#### model_1 = NN([7, 5, 1]), l_rate=1, after 200 epochs
was jumping around [[0.27610154]]
#### model_1 = NN([7, 5, 3, 1]), l_rate=2, epochs=10
J: [[0.51498522]] => [[0.51496249]]
MAE: [[0.33480384]] => [[0.33408482]]
Accuracy: 82885/105017 (78.93) => 82885/105017 (78.93)
'''

def softmax(arr):
  new_arr = []
  for el in arr:
    new_arr.append(np.exp(el)/np.exp(arr).sum())
  return new_arr


import numpy as np
import time
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
line, = ax.plot([], [])

def update_plot(cost_values):
    line.set_data(range(len(cost_values)), cost_values)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.1)

for i in range(100):
    print(i)
    cf_arr = np.arange(i)**2
    update_plot(cf_arr)
    time.sleep(1)

# plt.show()