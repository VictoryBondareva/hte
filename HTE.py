import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import defaultdict
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import truncnorm

def reader(file):
	with open(file) as f:
		input_data = pd.read_csv(f)
	return input_data

def validate_dataset(dataset, to_factirize):
	for column in to_factirize:
		dataset[column] = pd.factorize(dataset[column])[0]
	return dataset

class HTE:
	def __init__(self):
		# self.trees_data = np.zeros((kNN*len(self.treatment), len(self.treatment.columns) + 1))
		# self.tree_data = np.zeros((0, 5))
		# self.model =  DecisionTreeRegressor()
		self.model = RandomForestRegressor(n_estimators=10)

	def init_data(self, dataset, control_data = None, is_treatment = 'T'): # T - столбец в котором 1 - леченый, 0 - не леченый
		# !!!надо удалить T!!!
		if control_data != None:
			self.treatment = dataset
			self.control = control_data
		else:
			self.treatment = dataset.loc[dataset[is_treatment] == 1]
			self.control = dataset.loc[dataset[is_treatment] == 0]
		# plt.plot(self.treatment['Temperature'], self.treatment['Y'], 'ro')
		# plt.plot(self.control['Temperature'], self.control['Y'], 'bo')
		# plt.show()
		# print(self.treatment)
		# print(self.control)
	def dist(self, a, b, ignore_col = None):
		res = 0
		# print(a.index)
		# for col_a in a.name:
		# 	print(col_a)
		
		cols_a , cols_b = self.ignoring(a, b, ignore_col)
		# print(list(cols_a))
		for col_a, col_b in zip(cols_a, cols_b):
			res += (col_a - col_b)**2
		return math.sqrt(res)

	def ignoring(self, a, b, ignore_Y_name):
		res_a = []
		res_b = []

		cols_a = list(filter(lambda x: str(x) not in ignore_Y_name, a.index))
		cols_b = list(filter(lambda x: str(x) not in ignore_Y_name, b.index))

		for col_a in cols_a:
			res_a.append(a[col_a])
		for col_b in cols_b:
			res_b.append(b[col_b])
		return res_a, res_b

	def classify_kNN(self, kNN = 3, Y_name = 'Y', is_treatment = 'T'):
		res_distance = []
		res_y = []
		# res_group = []
		tmp_control = pd.DataFrame()
		self.tree_data = np.zeros((0, (len(self.control.columns) - 2)*2))
		self.tree_delta_y = []

		# print(tmp_control)
		for i, treatment_row in self.treatment.iterrows():
			for j, control_row in self.control.iterrows():
				res_distance.append(self.dist(treatment_row, control_row,  [Y_name, is_treatment]))
				res_y.append(treatment_row[Y_name] - control_row[Y_name])
			tmp_control['Distance'] = pd.Series(res_distance, index = self.control.index)
			tmp_control['delta_Y'] = pd.Series(res_y, index = self.control.index)
			# print(res_y)
			tmp_control = tmp_control.sort_values(by=['Distance'])
			# print('L tr: ', i, '\n', tmp_control)
			
			tmp_control.drop(tmp_control.index[kNN:], inplace = True)
			self.tree_delta_y += list(tmp_control['delta_Y'].values)
			# print(tmp_control)
			for k, tmp_control_row in tmp_control.iterrows():
				# print('tr: \n', self.treatment.loc[i], 'cl: \n', self.control.loc[k])
				tmp_tr, tmp_cl = self.ignoring(self.treatment.loc[i], self.control.loc[k], [Y_name, is_treatment])
				# print([list(tmp_tr) + list(tmp_cl)])
				self.tree_data = np.append(self.tree_data, [list(tmp_tr) + list(tmp_cl)], axis = 0)

			# print(self.tree_data)
			# print(self.tree_delta_y)

			res_y.clear()
			res_distance.clear()
			tmp_control = tmp_control.iloc[0:0]
			
	def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
		return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

	def modification_1(self, dataset, control_data = None, is_treatment = 'T', Y_name = 'Y', kNN = 3, alpha = 3):
		self.init_data(dataset, control_data = control_data, is_treatment = is_treatment)
		tree_mu0 = RandomForestRegressor(n_estimators=10)
		tree_mu0.fit(self.control.loc[self.control.columns != Y_name],self.control.loc[self.control.columns == Y_name])
		


	def fit(self, dataset, control_data = None, is_treatment = 'T', Y_name = 'Y', kNN = 3):
		self.init_data(dataset, control_data = control_data, is_treatment = is_treatment)
		self.classify_kNN(kNN = kNN, Y_name = Y_name, is_treatment = is_treatment)
		self.tree_delta_y = np.array(self.tree_delta_y)
		# print(self.tree_data, self.tree_delta_y)
		self.model.fit(self.tree_data, self.tree_delta_y)

	def predict(self, predicted):
		return self.model.predict(predicted)



# dataset = reader('for_HTE.txt')
# # datatrain.Date = pd.to_datetime(datatrain.Date)
# dataset = validate_dataset(dataset, ['Sex'])
# print(dataset)

# hte = HTE()
# hte.fit(dataset, is_treatment = 'T')
# print(hte.predict([
# 	[50., 1., 39.7, 50., 1., 39.7],
# 	[40., 1., 36.6, 40., 1., 36.6]
# 	]))

# # Create a random dataset
# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis=0)
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - rng.rand(16))

# # Fit regression model
# regr_1 = DecisionTreeRegressor(max_depth=2)
# regr_2 = DecisionTreeRegressor(max_depth=5)
# regr_1.fit(X, y)
# regr_2.fit(X, y)

# # Predict
# X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)

# # Plot the results
# plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black",
#             c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue",
#          label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.show()
