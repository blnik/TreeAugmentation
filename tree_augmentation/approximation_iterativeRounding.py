# working with graphs
import networkx as nx

# plotting 
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

from random import sample
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import time
from os.path import exists
from os import makedirs

# gurobi LP solver
import gurobipy as gp
from gurobipy import GRB
from tree_augmentation.general import *
from tree_augmentation.natural_lp import *
from tree_augmentation.approximation_uplinks import *
from tree_augmentation.approximation_naiveRounding import *

def iterative_rounding(LP, link_vars, link_weights):
    """
    Parameters
        LP:        optimised gurobi LP model
        link_vars: decision variables (i.e. variables corresponding to links) in given LP
    Returns
        approximate solution after running the iterative rounding procedure with the corresponding objective value.
        Note that this is a recursive function. 
        Note further that this function is only allowed to be called if the LP is optimal
    """
    naturalLP_solution = LP.getAttr('X')
    naturalLP_solution_np = np.array(naturalLP_solution)
    # if the solution of the natural LP relaxation is itelf binary, return the binary solution
    if is_binary(naturalLP_solution):
        approx_solution = naturalLP_solution
        approx_obj_value = LP.objVal
        return approx_solution, approx_obj_value
    
    # if some value of the optimal solution of the natural LP relaxation is in (0, 0.5), then 
    # a) add constraints to the natural LP relaxation (set all variables to 1 that have value >= 0.5)
    # b) optimise the newly obtained LP
    # c) call iterative_rounding on the optimised LP 
    elif np.any((naturalLP_solution_np > 0) & (naturalLP_solution_np < 0.5)):
        LP.addConstr(link_vars[naturalLP_solution_np >= 0.5] == 1.0)
        LP.optimize()
        return iterative_rounding(LP, link_vars, link_weights)
    
    # if solution is not binary and no variable lies strictly betwen 0 and 0.5, round up all variables
    else:
        approx_solution = np.ceil(naturalLP_solution_np)
        approx_obj_value = link_weights @ approx_solution
        return approx_solution, approx_obj_value
    
    
def approximation_via_iterativeRounding(tree, links, link_weights = None):
    """
    Parameters
        tree:          tree under consideration
        links:         list of links
        link_weights:  weights of each link in links (if None, we assume unit weights for each link)
    Returns
        approximation to the optimal solution obtained from iteratively rounding the LP solution:
        1. run the natural LP approximation
        2. if the optimal solution has values strictly between 0 and 0.5:
               set all variables with value >= 0.5 to 1 and go to step 1
           if the optimal solution is binary:
               return binary solution
           if the optimal solution ony has fractional components in [0.5,1):
               return solution where we round up all fractional components to 1
    """
    if link_weights is None:
        link_weights = np.ones(len(links))
        
    
    # set up natural LP
    naturalLP, link_vars = set_up_naturalLP(tree, links, link_weights = link_weights, variable_type = GRB.CONTINUOUS)
    
    
    # solve LP
    naturalLP.optimize()

    lp_status = decode_lp_status(naturalLP.getAttr('Status'))
    # if LP is optimal, run iterative rounding procedure as described above.
    if lp_status == 'opt':
        approx_solution, approx_obj_value = iterative_rounding(naturalLP, link_vars, link_weights)
    else:
        approx_solution = []
        approx_obj_value = None
    output = {
        'lp_status': lp_status,
        'optimal_solution': approx_solution,
        'optimal_obj_value': approx_obj_value
    }
    return output



def run_simulations_approximation_via_iterativeRounding(samples, n, factor, output_directory, disable_progress_bar = False):
    """
    Parameters
        samples:              number of TAP instances for which to run the algorithms
        n:                    number of nodes of the tree
        factor:               defines how many links to generate. We generate factor * (# of edges of tree) links.
        disable_progress_bar: True or False, whether or not to disable the progress bar
    Returns
        approx_objectives: array of the optimal objective values of the approximation algorithm via iterative rounding
        approx_solutions:  array of the optimal solutions of the approximation algorithm via iterative rounding
        exact_objectives:  array of the optimal objective values of the exact algorithm
        exact_solutions:   array of the optimal solutions of the exact algorithm
        We also save the corresponding arrays in the output_directory folder.
    """
    approx_objectives = []
    approx_solutions = []
    exact_objectives = []
    exact_solutions = []
    for k in tqdm(range(samples), disable = disable_progress_bar, desc=f'IRD: n = {n}, factor = {factor}'):
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
            approximation_output = approximation_via_iterativeRounding(tree = T, links = links)
            approx_objectives.append(approximation_output['optimal_obj_value'])
            approx_solutions.append(approximation_output['optimal_solution'])
            
    # save results in the output_directory file
    approx_objectives = np.array(approx_objectives, dtype = object)
    approx_solutions = np.array(approx_solutions, dtype = object)
    exact_objectives = np.array(exact_objectives, dtype = object)
    exact_solutions = np.array(exact_solutions, dtype = object)
    output_filepath = f'{output_directory}/iterativeRounding_approximation/samples{samples}/n{n}/'
    create_directory(output_filepath)
    filename_prefix = f'samples{samples}_n{n}_factor{factor}_'
    np.save(f'{output_filepath}{filename_prefix}approx_objectives.npy', approx_objectives)
    np.save(f'{output_filepath}{filename_prefix}approx_solutions.npy', approx_solutions)
    np.save(f'{output_filepath}{filename_prefix}exact_objectives.npy', exact_objectives)
    np.save(f'{output_filepath}{filename_prefix}exact_solutions.npy', exact_solutions)
    return approx_objectives, approx_solutions, exact_objectives, exact_solutions



def get_approximation_ratios_of_iterativeRounding_approximation(samples, n, factor, output_directory):
    """
    Parameters
        samples: number of TAP instances for which to compute the approximation ratio
        n:       number of nodes of the tree in the TAP instance
        factor:  defines how many links to generate. We generate factor * (# of edges of tree) links.
    Returns
        array storing the approximation ratios of the iterative rounding approximation algorithm 
        for each of the (samples) TAP instances
    """
    file_path = f'{output_directory}/iterativeRounding_approximation/samples{samples}/n{n}/'
    filename_prefix = f'samples{samples}_n{n}_factor{factor}_'
    fp_eo = f'{file_path}{filename_prefix}exact_objectives.npy'
    fp_ao = f'{file_path}{filename_prefix}approx_objectives.npy'
    
    # if the TAP instances have not yet been computed, compute them
    if (not exists(fp_eo)) or (not exists(fp_ao)):
        approx_objectives,_,exact_objectives,_ = run_simulations_approximation_via_iterativeRounding(
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
