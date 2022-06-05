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

from tree_augmentation.general import *
# gurobi LP solver
import gurobipy as gp
from gurobipy import GRB

from tree_augmentation.general import *
from tree_augmentation.natural_lp import *



def transform_to_uplinks(tree, root, links, link_weights = None):
    """
    Parameters
        tree:         tree under consideration
        root:         root of tree
        links:        list of links
        link_weights: list storing the weights of the original links
    Returns
        transformed_links:        list of links which contains only uplinks. The transformation is based on 
                                  Adjiashvili 2016. If a link (u,v) is not an uplink, then it is replaced by the 
                                  two links (u,w) and (v,w) where w is the nearest common ancester of u and v.
        transformed_link_weights: list of weights of the transformed links, where the weight is equal to the weight
                                  of the link from which the transformed link was created.
    """
    if link_weights is None:
        link_weights = np.ones(len(links))
        
    ncas = find_nca(tree, root, links)
    link_types = check_link_types(links = links, ncas = ncas, root = root)

    transformed_links = []
    transformed_link_weights = []
    for i in range(len(links)):
        # check if link is uplink. If yes, perform replacement as described above.
        if link_types[i] != 'u':
            link = links[i]
            link_endpoint_1 = link[0]
            link_endpoint_2 = link[1]
            link_nca = ncas[i]
            transformed_links.append(create_edge(link_endpoint_1, link_nca))
            transformed_links.append(create_edge(link_endpoint_2, link_nca))
            # both transformed links have the same weight as link
            transformed_link_weights.append(link_weights[i])
            transformed_link_weights.append(link_weights[i])
        else:
            transformed_links.append(links[i])
            transformed_link_weights.append(link_weights[i])
    return transformed_links, np.array(transformed_link_weights)


def transform_uplink_solution_to_link_solution(uplinks, links, uplink_solution):
    """
    Parameters
        uplinks:         list of uplinks obtained from links
        links:           list of links
        uplink_solution: solution to LP relaxation on uplinks
    Returns
        list storing the solution corresponding to the original links. It is obtained as follows:
        for each element l of links, if l is an uplink, then get the solution of the corresponding 
        link in the uplinks list; if l is not an uplink, then the uplinks list contains two uplinks 
        which correspond to l. By Adjiashvili 2016, if at least one of them has value 1, 
        we take the value of l to be 1 and zero otherwise.
    """
    link_solution = []
    uplink_index = 0
    for link in links:
        # if link is an uplink, take the value from uplink_solution
        if uplinks[uplink_index] == link:
            link_solution.append(uplink_solution[uplink_index])
            uplink_index += 1
        # if link is not an uplink...
        else:
            # if both uplinks corresponding to link have value 0, set the value of link to 0
            if uplink_solution[uplink_index] == 0 and uplink_solution[uplink_index+1] == 0:
                link_solution.append(0.0)
            # if at least one of the two uplinks corresponding to link has value 1, set the value of link to 1.
            else:
                link_solution.append(1.0)
            # in this case, 2 elements in the uplinks list correspond to the current link. 
            # We hence must increase the index by 2
            uplink_index += 2
    return link_solution



def approximation_via_uplinks(tree, root, links, link_weights = None):
    """
    Parameters
        tree:          tree under consideration
        root:          root of the tree
        links:         list of links
        link_weights:  weights of each link in links (if None, we assume unit weights for each link)
    Returns
        approximation to the optimal solution obtained from replacing each cross- or inlink by two uplinks 
        and then solving the natural LP relaxation (the procedure is from Adjiashvili 2016).
    """
    # obtain list of uplinks from the given list of links
    uplinks, uplinks_weights = transform_to_uplinks(tree, root, links, link_weights)
    # solve the natural LP relaxation using only the uplinks
    uplink_LP_output = solve_naturalLP(tree, uplinks, uplinks_weights, variable_type = GRB.CONTINUOUS)
    
    if link_weights is None:
        link_weights = np.ones(len(links))

    if uplink_LP_output['lp_status'] == 'opt':
        uplink_solution = uplink_LP_output['optimal_solution']
        # from the optimal solution for the only-uplink-instance, 
        # obtain an approximate solution for the original links
        link_solution = transform_uplink_solution_to_link_solution(uplinks, links, uplink_solution)
        obj_value = link_weights @ np.array(link_solution)
    else:
        link_solution = []
        obj_value = None
    output = {
        'lp_status': uplink_LP_output['lp_status'],
        'optimal_solution': link_solution,
        'optimal_obj_value': obj_value
    }
    return output



def run_simulations_approximation_via_uplinks(samples, n, factor, output_directory, disable_progress_bar = False):
    """
    Parameters
        samples:              number of TAP instances for which to run the algorithms
        n:                    number of nodes of the tree
        factor:               defines how many links to generate. We generate factor * (# of edges of tree) links.
        output_directory:     directory in which to save the results
        disable_progress_bar: True or False, whether or not to disable the progress bar
    Returns
        approx_objectives: array of the optimal objective values of the approximation algorithm via uplinks
        approx_solutions:  array of the optimal solutions of the approximation algorithm via uplinks
        exact_objectives:  array of the optimal objective values of the exact algorithm
        exact_solutions:   array of the optimal solutions of the exact algorithm
        We also save the corresponding arrays in the output_directory folder.
    """
    approx_objectives = []
    approx_solutions = []
    exact_objectives = []
    exact_solutions = []
    for k in tqdm(range(samples), disable = disable_progress_bar, desc = f'UPL: n = {n}, factor = {factor}'):
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
            
            # compute the approximate solution using the uplinks approximation algorithm
            approximation_output = approximation_via_uplinks(tree = T, root = 1, links = links)
            approx_objectives.append(approximation_output['optimal_obj_value'])
            approx_solutions.append(approximation_output['optimal_solution'])
            
    # save results in the output_directory file
    approx_objectives = np.array(approx_objectives, dtype = object)
    approx_solutions = np.array(approx_solutions, dtype = object)
    exact_objectives = np.array(exact_objectives, dtype = object)
    exact_solutions = np.array(exact_solutions, dtype = object)
    
    output_filepath = f'{output_directory}/uplink_approximation/samples{samples}/n{n}/'
    create_directory(output_filepath)
    filename_prefix = f'samples{samples}_n{n}_factor{factor}_'
    np.save(f'{output_filepath}{filename_prefix}approx_objectives.npy', approx_objectives)
    np.save(f'{output_filepath}{filename_prefix}approx_solutions.npy', approx_solutions)
    np.save(f'{output_filepath}{filename_prefix}exact_objectives.npy', exact_objectives)
    np.save(f'{output_filepath}{filename_prefix}exact_solutions.npy', exact_solutions)
    return approx_objectives, approx_solutions, exact_objectives, exact_solutions


def get_approximation_ratios_of_uplinks_approximation(samples, n, factor, output_directory):
    """
    Parameters
        samples: number of TAP instances for which to compute the approximation ratio
        n:       number of nodes of the tree in the TAP instance
        factor:  defines how many links to generate. We generate factor * (# of edges of tree) links.
    Returns
        array storing the approximation ratios of the uplink approximation algorithm for each TAP instance
    """
    file_path = f'{output_directory}/uplink_approximation/samples{samples}/n{n}/'
    filename_prefix = f'samples{samples}_n{n}_factor{factor}_'
    fp_ao = f'{file_path}{filename_prefix}approx_objectives.npy'
    fp_eo = f'{file_path}{filename_prefix}exact_objectives.npy'
    
    # if the TAP instances have not yet been computed, compute them
    if (not exists(fp_ao)) or (not exists(fp_eo)):
        approx_objectives, _, exact_objectives, _ = run_simulations_approximation_via_uplinks(
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


