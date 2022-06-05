# working with graphs
import networkx as nx

# plotting 
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

from random import sample, seed
import numpy as np
from tqdm import tqdm
import time
from os.path import exists
from os import makedirs

# gurobi LP solver
import gurobipy as gp
from gurobipy import GRB
env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

from tree_augmentation.general import *


def create_edge_link_cover_matrix(tree, links, tree_edges, number_edges, number_links):
    """
    Parameters
        tree:         tree under consideration
        links:        list of links
        tree_edges:   edges of the tree stored in a list
        number_edges: number of edges in the tree
        number_links: number of links in the tree
    Returns
        edge-link cover matrix C which is a matrix of dimension number_edges x number_links with binary entries 
        where C[i,j] = 1 if link j covers link i and 0 otherwise.
    """
    tree_edges_clean = [create_edge(e[0], e[1]) for e in tree_edges]
    tree_edges = tree_edges_clean
    edge_link_cover_matrix = np.zeros([number_edges, number_links])
    
    # for each link (u,v), find the shortest path P from u to v in tree and set all entries in edge_link_cover_matrix
    # corresponding to the edges in P to 1. Note that since we work in trees, P is always unique.
    for link_index, link in enumerate(links):
        shortest_path = nx.shortest_path(tree, link[0], link[1])
        shortest_path_length = len(shortest_path)
        for i in range(shortest_path_length - 1):
            edge_in_shortest_path = create_edge(shortest_path[i], shortest_path[i+1])
            edge_index = tree_edges.index(edge_in_shortest_path)     # find index of the edge in the original tree.
            edge_link_cover_matrix[edge_index, link_index] = 1
    return edge_link_cover_matrix


def decode_lp_status(encoded_lp_status):
    """
    Parameters
        encoded_lp_status: LP status as returned by gurobi (int 1-16)
    Returns
        LP status as string (opt for optimal, inf for infeasible, error for everything else).
    """
    if encoded_lp_status == 2:
        return 'opt'
    elif encoded_lp_status == 3:
        return 'inf'
    else:
        return 'error'
    
    

    
    
    
def set_up_naturalLP(tree, links, link_weights = None, variable_type = GRB.BINARY):
    """
    Parameters
        tree:          tree under consideration
        links:         list of links
        link_weights:  weights of each link in list links (if None, we assume unit weights for each link)
        variable_type: (GRB.BINARY or GRB.CONTINUOUS) specifies whether to solve LP relaxation or integer LP.
    Returns
        naturalLP: a gurobi LP model of the natural LP (relaxation if variable_type = GRB.CONTINUOUS) 
        link_vars: the decision variables of the model (i.e. an array of variables each corresponding to a link).
    """
    tree_edges = list(tree.edges())
    number_edges = len(tree_edges)
    number_links = len(links)

    # construct edge_link_cover_matrix
    edge_link_cover_matrix = create_edge_link_cover_matrix(tree, links, tree_edges, number_edges, number_links)

    # construct natural LP
    naturalLP = gp.Model('naturalLP', env = env)
    link_vars = naturalLP.addMVar(shape = number_links, vtype = variable_type, name = 'link_vars', lb = 0, ub = 1)
    if link_weights is None:
        link_weights = np.ones(number_links)
    naturalLP.setObjective(link_weights @ link_vars, GRB.MINIMIZE)
    naturalLP.addConstr(edge_link_cover_matrix @ link_vars >= 1, name = 'c')
    
    return naturalLP, link_vars



def solve_naturalLP(tree, links, link_weights = None, variable_type = GRB.BINARY):
    """
    Parameters
        tree:          tree under consideration
        links:         list of links
        link_weights:  weights of each link in list links (if None, we assume unit weights for each link)
        variable_type: (GRB.BINARY or GRB.CONTINUOUS) specifies whether to solve LP relaxation or integer LP.
    Returns
        dict with the following keys
        lp_status: LP status of the optimised LP (opt for optimal, inf for infeasible, error for everything else)
        optimal_solution:  list of optimal values of decision variables (same order as links) 
                           (empty list if no optimal solution is found)
        optimal_obj_value: objective value corresponding to the optimal solution 
                           (None if no optimal solution is found)
    """
    # set up natural LP
    naturalLP, link_vars = set_up_naturalLP(tree, links, link_weights, variable_type)
    
    # solve LP
    naturalLP.optimize()
    
    # format output
    lp_status = decode_lp_status(naturalLP.getAttr('Status'))
    if lp_status == 'opt':
        optimal_solution = naturalLP.getAttr('X')
        optimal_obj_value = naturalLP.objVal
    else:
        optimal_solution = []
        optimal_obj_value = None
    output = {
        'lp_status': lp_status, 
        'optimal_solution': optimal_solution, 
        'optimal_obj_value': optimal_obj_value
    }
    naturalLP.write('naturalLP.attr')
    return output
    
    
    
#def solve_naturalLP(tree, links, link_weights = None, variable_type = GRB.BINARY):
#    """
#    Parameters
#        tree:          tree under consideration
#        links:         list of links
#        link_weights:  weights of each link in list links (if None, we assume unit weights for each link)
#        variable_type: (GRB.BINARY or GRB.CONTINUOUS) specifies whether to solve LP relaxation or integer LP.
#    Returns
#        dict with the following keys
#        lp_status: LP status of the optimised LP (opt for optimal, inf for infeasible, error for everything else)
#        optimal_solution:  list of optimal values of decision variables (same order as links) 
#                           (empty list if no optimal solution is found)
#        optimal_obj_value: objective value corresponding to the optimal solution 
#                           (None if no optimal solution is found)
#    """
#    tree_edges = list(tree.edges())
#    number_edges = len(tree_edges)
#    number_links = len(links)
#    
#    # construct edge_link_cover_matrix
#    edge_link_cover_matrix = create_edge_link_cover_matrix(tree, links, tree_edges, number_edges, number_links)
#
#    # construct natural LP
#    naturalLP = gp.Model('naturalLP', env = env)
#    link_vars = naturalLP.addMVar(shape = number_links, vtype = variable_type, name = 'link_vars', lb = 0, ub = 1)
#    if link_weights is None:
#        link_weights = np.ones(number_links)
#    naturalLP.setObjective(link_weights @ link_vars, GRB.MINIMIZE)
#    naturalLP.addConstr(edge_link_cover_matrix @ link_vars >= 1, name = 'c')
#    
#    # solve LP
#    naturalLP.optimize()
#    
#    # format output
#    lp_status = decode_lp_status(naturalLP.getAttr('Status'))
#    if lp_status == 'opt':
#        optimal_solution = naturalLP.getAttr('X')
#        optimal_obj_value = naturalLP.objVal
#    else:
#        optimal_solution = []
#        optimal_obj_value = None
#    output = {
#        'lp_status': lp_status, 
#        'optimal_solution': optimal_solution, 
#        'optimal_obj_value': optimal_obj_value
#    }
#    naturalLP.write('naturalLP.attr')
#    return output




def is_binary(l):
    """
    Parameters
        l: list of numbers
    Returns
        True if all elements of l is 0 or 1, and False otherwise
    """
    array = np.array(l)
    return np.all(np.isin(array, [0,1]))

def is_fully_fractional(l):
    """
    Parameters
        l: list of numbers
    Returns
        True if all elements of l are fractional, and False otherwise
    """
    array = np.array(l)
    return ~np.any(np.isin(array, [1]))