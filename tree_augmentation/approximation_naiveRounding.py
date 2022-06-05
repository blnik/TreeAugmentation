# working with graphs
import networkx as nx

# plotting 
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

from random import sample
import numpy as np
from tqdm.notebook import tqdm
import time
from os.path import exists
from os import makedirs

# gurobi LP solver
import gurobipy as gp
from gurobipy import GRB

from tree_augmentation.natural_lp import *


def approximation_via_naiveRounding(tree, links, link_weights = None):
    """
    Parameters
        tree:          tree under consideration
        links:         list of links
        link_weights:  weights of each link in links (if None, we assume unit weights for each link)
    Returns
        approximation to the optimal solution obtained from 
        ceiling the optimal solution of the natural LP relaxation.
    """
    
    if link_weights is None:
        link_weights = np.ones(len(links))
    
    naturalLP_output = solve_naturalLP(tree, links, link_weights, variable_type = GRB.CONTINUOUS)
    if naturalLP_output['lp_status'] != 'opt':
        approx_solution = []
        approx_obj_value = None
    else:
        naturalLP_solution = naturalLP_output['optimal_solution']
        naturalLP_solution = np.array(naturalLP_solution)
        approx_solution = np.ceil(naturalLP_solution)
        approx_obj_value = approx_solution @ link_weights
    
    
    output = {
        'lp_status': naturalLP_output['lp_status'],
        'optimal_solution': approx_solution,
        'optimal_obj_value': approx_obj_value
    }
    return output

def run_simulations_approximation_via_naiveRounding(samples, n, factor, output_directory, disable_progress_bar = False):
    """
    Parameters
        samples:              number of TAP instances for which to run the algorithms
        n:                    number of nodes of the tree
        factor:               defines how many links to generate. We generate factor * (# of edges of tree) links.
        disable_progress_bar: True or False, whether or not to disable the progress bar
    Returns
        approx_objectives: array of the optimal objective values of the approximation algorithm via naive rounding
        approx_solutions:  array of the optimal solutions of the approximation algorithm via naive rounding
        exact_objectives:  array of the optimal objective values of the exact algorithm
        exact_solutions:   array of the optimal solutions of the exact algorithm
        We also save the corresponding arrays in the output_directory folder.
    """
    approx_objectives = []
    approx_solutions = []
    exact_objectives = []
    exact_solutions = []
    for k in tqdm(range(samples), disable = disable_progress_bar, desc = f'NRD: n = {n}, factor = {factor}'):
        # generate TAP instance
        T = nx.random_tree(n, seed = k)
        if factor == 'allLinks':
            number_links = int(n*(n-1)/2)
        else:
            number_links = factor * len(T.edges())
        links = generate_random_links_by_type(T, 1, ['c', 'i', 'u'], number_links, seed = k+1)
        
        # compute exact solution
        LP_output = solve_naturalLP(tree = T, links = links, variable_type = GRB.BINARY)
        # if exact solution is infeasible, then approximation is also infeasible
        if LP_output['lp_status'] != 'opt':
            approx_objectives.append(None)
            approx_solutions.append([])
            exact_objectives.append(None)
            exact_solutions.append([])
        # if exact solution is optimal, compute the approximate solution
        else:
            exact_objectives.append(LP_output['optimal_obj_value'])
            exact_solutions.append(LP_output['optimal_solution'])
            
            # compute the approximate solution using the naive rounding approximation algorithm
            approximation_output = approximation_via_naiveRounding(tree = T, links = links)
            approx_objectives.append(approximation_output['optimal_obj_value'])
            approx_solutions.append(approximation_output['optimal_solution'])
            
    # save results in the OUTPUT_DIRECTORY file
    approx_objectives = np.array(approx_objectives, dtype = object)
    approx_solutions = np.array(approx_solutions, dtype = object)
    exact_objectives = np.array(exact_objectives, dtype = object)
    exact_solutions = np.array(exact_solutions, dtype = object)
    output_filepath = f'{output_directory}/naiveRounding_approximation/samples{samples}/n{n}/'
    create_directory(output_filepath)
    filename_prefix = f'samples{samples}_n{n}_factor{factor}_'
    np.save(f'{output_filepath}{filename_prefix}approx_objectives.npy', approx_objectives)
    np.save(f'{output_filepath}{filename_prefix}approx_solutions.npy', approx_solutions)
    np.save(f'{output_filepath}{filename_prefix}exact_objectives.npy', exact_objectives)
    np.save(f'{output_filepath}{filename_prefix}exact_solutions.npy', exact_solutions)
    return approx_objectives, approx_solutions, exact_objectives, exact_solutions



def get_approximation_ratios_of_naiveRounding_approximation(samples, n, factor, output_directory):
    """
    Parameters
        samples: number of TAP instances for which to compute the approximation ratio
        n:       number of nodes of the tree in the TAP instance
        factor:  defines how many links to generate. We generate factor * (# of edges of tree) links.
    Returns
        array storing the approximation ratios of the naive rounding approximation algorithm for each TAP instance
    """
    file_path = f'{output_directory}/naiveRounding_approximation/samples{samples}/n{n}/'
    filename_prefix = f'samples{samples}_n{n}_factor{factor}_'
    fp_eo = f'{file_path}{filename_prefix}exact_objectives.npy'
    fp_ao = f'{file_path}{filename_prefix}approx_objectives.npy'
    
    # if the TAP instances have not yet been computed, compute them
    if (not exists(fp_eo)) or (not exists(fp_ao)):
        approx_objectives,_,exact_objectives,_ = run_simulations_approximation_via_naiveRounding(
            samples, n, factor, output_directory
        )
    # if the TAP instances have been computed, load them from the outputs directory.
    else:
        approx_objectives = np.load(fp_ao, allow_pickle = True)
        exact_objectives = np.load(fp_eo, allow_pickle = True)
    
    
    approximation_ratios = np.empty(len(approx_objectives))
    
    # set all approximation ratios which correspond to infeasible instances to None
    approximation_ratios[(approx_objectives == None) | (exact_objectives == None)] = np.nan
    
    # for all feasible instances (i.e. where the objectives are not None), compute the approximation ratio
    approx_objectives_feasible = approx_objectives[approx_objectives != None]
    exact_objectives_feasible = exact_objectives[exact_objectives != None]
    approximation_ratios[(approx_objectives != None) & (exact_objectives != None)] = approx_objectives_feasible / exact_objectives_feasible
    
    return approximation_ratios