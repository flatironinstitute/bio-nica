#!/usr/bin/env python

import argparse
import code # for code.interact(local=dict(globals(), **locals()) ) debugging
import glob
import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from quadprog import solve_qp
import scipy.linalg
from scipy.optimize import linear_sum_assignment
import time

from typing import Dict, List, Tuple # for mypy

######################
# bionica - comparison of NICA algorithms
#
# Reproduces experiments from the paper:
# "A biologically plausible single-layer network for blind nonnegative source separation"

def main():
	cfg = handle_args()
	y_dim = s_dim
	A, S, X = load_data(cfg)
	if np.algorithm == 'bio_nica':
		dispatch_bio_nica(cfg, y_dim, A, S, X)
	elif np.algorithm == '2nsm':
		dispatch_nsm(cfg, y_dim, A, S, X)
	elif np.algorithm == 'nn_pca':
		dispatch_nnpca(cfg, y_dim, A, S, X)
	elif np.algorithm == 'fast_ica':
		dispatch_fastica(cfg, y_dim, A, S, X)


def handle_args():
	parser = argparse.ArgumentParser(description="This does a thing")
	parser.add_argument("--s_dim",	type=int, default=3, help="Source dimension")
	parser.add_argument("--x_dim",	type=int, default=3, help="Mixed Stimuli Dimension, must satisfy x_dim >= s_dim")
	parser.add_argument("--trials",	type=int, default=1, help="What does this do?")
	parser.add_argument("--epochs",	type=int, default=1, help="What does this do?")
	parser.add_argument("--image_data",	type=bool, action='store_true', help="What does this do?")

	parser.add_argument("--algorithm", required=True, choices=['bio_nica', '2nsm', 'nn_pca', 'fast_ica'], help="Which algorithm to run. Valid: bio_nica, 2nsm, nn_pca, fast_ica")

	ret = parser.parse_args()
	return ret

######################
# Dispatch functions. Each of these performs an algorithm over the loaded data, saving the error and
# runtime information for later analysis.

def dispatch_bio_nica(cfg, y_dim, A, S, X) -> None:
	pass

def dispatch_nsm(cfg, y_dim, A, S, X) -> None:
	pass

def dispatch_nnpca(cfg, y_dim, A, S, X) -> None:
	pass

def dispatch_fastica(cfg, y_dim, A, S, X) -> None:
	pass

######################

def load_data(cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	""" Loads the datasets for the comparison; source path depends on the --image_Data flag """
	if cfg.image_data:
		path_prefix = os.path.join(['data', 'images'])
	else:
		path_prefix = os.path.join(['data', f'{cfg.s_dim}-dim-source'])
	A = np.load(os.path.join([path_prefix, 'mixing-matrix.npy']))
	S = np.load(os.path.join([path_prefix, 'sources.npy'      ]))
	X = np.load(os.path.join([path_prefix, 'mixtures.npy'     ]))
	return A, S, X

#####
main()
