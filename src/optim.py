from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import os
import sys
from io import StringIO
import numpy as np
import pandas as pd
import csv
from model.config import Config
from train import main as m_train
from test import main as m_test


class Capturing(list):
	def __enter__(self):
		self._stdout = sys.stdout
		sys.stdout = self._stringio = StringIO()
		return self

	def __exit__(self, *args):
		self.extend(self._stringio.getvalue().splitlines())
		del self._stringio  # free up some memory
		sys.stdout = self._stdout


class Optimizer:

	def __init__(self, optimization_folder, nr_iterations, iteration_chunck_size, nr_init_points, embedder='LASEREmbedderI', log_file_name = 'logs.json', load_log=False, prev_log='log.json', probe=True):

		# Set static variables
		self.INTERMEDIATE_RESULTS_FOLDER = optimization_folder
		self.FINAL_RESULTS_FOLDER = optimization_folder
		self.NR_ITERATIONS = nr_iterations
		self.ITERATION_CHUNCK_SIZE = iteration_chunck_size
		self.NR_INIT_POINTS = nr_init_points
		self.EMBEDDER = embedder
		self.probe = probe
		self.config = Config()
		self.log_file = os.path.join(optimization_folder, log_file_name)
		self.prev_log = os.path.join(optimization_folder, prev_log)
		self.logger = JSONLogger(path=self.log_file)
		self.load_log = load_log
		# Boundaries between which to explore the input space
		self.param_boundaries = {
			'dropout_before_laser':(0., 0.5),
			'dropout_in_laser':(0., 0.5),
			'transformer_drop': (0., 0.5),
			'dropout':(0., 0.5),
			'hidden_size_lstm': (50, 350),
			'weight_decay': (0., 0.1),
			'learning_rate_warmup_steps': (1., 10.0),
			'num_heads':(0.5, 4.49),
			'filter_size': (3.5, 350)
		}
		# Set points on which to evaluate the model for exploration of the solution space
		self.explore_points = {
			'dropout_before_laser':[0.1],
			'dropout_in_laser':[0.25],
			'transformer_drop': [0.4],
			'dropout':[0.5],
			'hidden_size_lstm': [300],
			'weight_decay': [0.01],
			'learning_rate_warmup_steps': [2.],
			'num_heads': [2.],
			'filter_size': [25.]
		}

		self.bo = None  # initialize variable for further error handling

		assert len(np.unique([len(n) for n in
							  self.explore_points.values()])) == 1, 'number of explore points should be the same for all parameters'
		self.NUM_EXPLORE_POINTS = np.unique([len(n) for n in self.explore_points.values()])[0]

	######################################################
	# Helper functions
	######################################################

	def _cast_params(self, params):
		# convert integers
		for param in ['hidden_size_lstm', 'learning_rate_warmup_steps', 'filter_size', 'num_heads']:
			params[param] = int(round(params[param]))

		if params['hidden_size_lstm'] % params['num_heads'] != 0:
			params['hidden_size_lstm'] -= params['hidden_size_lstm'] % params['num_heads']
		# convert integers
		for param in ['hidden_size_lstm', 'learning_rate_warmup_steps', 'filter_size', 'num_heads']:
			params[param] = int(round(params[param]))

		return params

	def _eval_fun(self, dropout_before_laser, dropout_in_laser, transformer_drop, dropout, hidden_size_lstm, weight_decay, learning_rate_warmup_steps, num_heads, filter_size):

		# TODO: Include default settings
		# params to evaluate:
		params = {
			'dropout_before_laser':dropout_before_laser,
			'dropout_in_laser':dropout_in_laser,
			'transformer_drop': transformer_drop,
			'dropout':dropout,
			'hidden_size_lstm': hidden_size_lstm,
			'weight_decay': weight_decay,
			'learning_rate_warmup_steps': learning_rate_warmup_steps,
			'num_heads': num_heads,
			'filter_size': filter_size
		}

		params = self._cast_params(params)
		self.config.set_manual_params(**params)
		# train the model
		m_train(self.config, embedders_to_train=[self.EMBEDDER])
		results, _ = m_test(self.config)
		score = np.mean(
			[results[os.path.join('parsed_data_lowercased', '{}_test_bio_bpe1.txt'.format(lang))][self.EMBEDDER]
			 for lang in ['eng'] #, 'ned', 'ger'] #, 'esp']
])

		return score #results[os.path.join('parsed_data_lowercased', 'eng_test_bio_bpe.txt')][self.EMBEDDER]

	#############################################################################
	# Public functions
	#############################################################################

	def init_csv(self, path):
		"""
			Initializes KNOWN target values stored in *.csv format to optimizer object
		"""

		df = pd.read_csv(path)
		self.bo.initialize_df(df)

	def get_best_params(self):
		"""
			Return the best found parameter settings after optimizing in dictionary form
		"""
		assert self.bo is not None, "Parameters can only be retrieved AFTER optimizing"

		params = self.bo.max['params']
		params = self._cast_params(params)

		return params

	# def _save_results(self, params, file_path):
	# 	with open(file_path, 'w', newline='') as csv_file:
	# 		writer = csv.writer(csv_file)
	# 		if isinstance(params, list):  # multiple intermediate param sets are stored in a list
	# 			for param_set in params:
	# 				for key, value in param_set.items():
	# 					writer.writerow([key, value])
	# 		elif isinstance(params, dict):  # opt params are saved in one dict
	# 			for key, value in params.items():
	# 				writer.writerow([key, value])

	def _explore_target_space(self):
		"""explore specific points before the gaussian process takes over
		exploration points are defined in self.explore_points"""
		for i in range(0, self.NUM_EXPLORE_POINTS):
			sub_dict = {}
			for key, value_list in self.explore_points.items():
				sub_dict[key] = value_list[i]
			self.bo.probe(
				params=sub_dict,
				lazy=True
			)
			self.bo.maximize(init_points=0, n_iter=0, acq='ei')

	def optimize(self):
		"""
		Main function for optimization
		"""
		# Initialize optimizer
		self.bo = BayesianOptimization(self._eval_fun, self.param_boundaries)
		self.bo.subscribe(Events.OPTMIZATION_STEP, self.logger)
		if self.load_log:
			load_logs(self.bo, logs=[self.prev_log])
		# Explore the input and target space on predefined points
		if self.probe:
			self._explore_target_space()

		# Set parameters for Gaussian Process
		gp_params = {}
		# {'kernel': None, 'alpha': 1e-5}
		self.bo.maximize(init_points=self.NR_INIT_POINTS, n_iter=self.NR_ITERATIONS, acq='ei', **gp_params)


		# # Iteratively maximize, store values, reinitialize
		# for i in range(0, self.NR_ITERATIONS, self.ITERATION_CHUNCK_SIZE):
		# 	# Only initialize extra random points in first epoch
		# 	nr_init_points = self.NR_INIT_POINTS if i == 0 else 1
		#
		# 	# Perform maximization
		# 	self.bo.maximize(init_points=nr_init_points, n_iter=self.ITERATION_CHUNCK_SIZE, acq='ei', **gp_params)
		#
		# 	int_file_path = os.path.join('optimizer', self.INTERMEDIATE_RESULTS_FOLDER, 'exploration.csv')
		# 	intermediate_results = self.bo.res
		# 	self._save_results(intermediate_results, file_path=int_file_path)
		#
		# # Store best results
		# final_file_path = os.path.join('optimizer', self.FINAL_RESULTS_FOLDER, 'final_optimization_results.csv')
		# final_results = self.bo.max
		# self._save_results(final_results, file_path=final_file_path)
		#
		# return final_file_path


if __name__ == "__main__":

	optimizer = Optimizer('optimizer', 25, 3, 0, embedder='LASEREmbedderI', load_log=True, log_file_name='pos_logs_eng_eng.json', probe=False, prev_log='pos_logs_eng_eng_old.json')
	optimizer.optimize()