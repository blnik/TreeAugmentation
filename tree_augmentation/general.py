# working with graphs
import networkx as nx

# plotting 
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

from random import sample
from random import seed as fix_seed
#import numpy as np
#from tqdm import tqdm
#import time
from os.path import exists
from os import makedirs



def create_edge(endpoint_1, endpoint_2):
    """
    Parameters
        endpoint_1: first endpoint of the edge we wish to create
        endpoint_2: second endpoint of the edge we wish to create
    Returns
        an edge (u,v) with the specified endpoints such that u < v.
    """
    tail = min(endpoint_1, endpoint_2)
    head = max(endpoint_1, endpoint_2)
    return (tail, head)


def all_links(number_nodes):
    """
    Parameters
        number_nodes: number of nodes
    Returns
        all possible links in a tree consisting of number_nodes nodes.
    """
    possible_links = []
    for i in range(number_nodes):
        for j in range(i+1, number_nodes):
            possible_links.append((i,j))
    return possible_links


def draw_tree(tree, root, links = None):
    """
    Parameters
        tree: tree under consideration
        root: root of the tree
    Outputs
        a plot of tree where root is located on the top
    """
    directed_tree = nx.bfs_tree(tree, root)
    pos = graphviz_layout(directed_tree, prog="dot")
    
    # plot the original tree
    nx.draw_networkx(
        directed_tree,
        pos = pos,
        edge_color = 'black',
        node_color = 'grey',
        arrowstyle = '-',
        width = 2
    )

    # plot the links
    if links != None:
        nx.draw_networkx_edges(
            directed_tree, 
            pos = pos, 
            edgelist = links,  
            edge_color = 'C0',
            connectionstyle='arc3,rad=0.6', 
            arrowstyle = '-',
            style = ':'
        )
    
    plt.axis('off')
    plt.show()
    
    
def find_nca(tree, root, links):
    """
    Parameters
        tree:  tree under consideration
        root:  root of the tree
        links: list of links
    Returns:
        list containing the nearest common ancestor for each of the links in the given tree with the given root
    """
    # construct a directed tree (needed for the tree_all_pairs_lowest_common_ancestor function)
    directed_tree = nx.bfs_tree(tree, root)
    # compute the nearest common ancestors for each link
    raw_ncas_generator = nx.tree_all_pairs_lowest_common_ancestor(directed_tree, root = root, pairs = links)
    
    # store nearest common ancestors in a list indexed in the same way as the links list.
    raw_ncas_dict = dict(raw_ncas_generator)
    #print(raw_ncas_list)
    ncas_clean = []
    for link in links:
        ncas_clean.append(raw_ncas_dict[link])
    return ncas_clean


def check_link_types(links, ncas, root):
    """
    Parameters
        links: list of links
        ncas:  list containing the nearest common ancesters for each link
        root:  root of the tree
    Returns
        list containing the link type for each of the links, where
        u = uplink
        i = inlink
        c = crosslink
    """
    link_types = []
    for i in range(len(links)):
        if ncas[i] in links[i]:
            link_types.append('u')
        elif ncas[i] == root:
            link_types.append('c')
        else:
            link_types.append('i')
    return link_types


def filter_links_by_type(links, link_types, types_to_include):
    """
    Parameters
        links:            list of links
        link_types:       list containing the type of links in the list links
        types_to_include: list of link types that we wish to filter by (u = uplink, i = inlink, c = crosslink)
    Returns
        a subset of links whose type is included in the types_to_include list
    """
    allowed_links = []
    for i in range(len(links)):
        if link_types[i] in types_to_include:
            allowed_links.append(links[i])
    return allowed_links



def generate_random_links_by_type(tree, root, types_to_include, number_links, seed = 0):
    """
    Parameters
        tree:             tree under consideration
        root:             root of the tree
        types_to_include: list of link types that we wish to filter by (u = uplink, i = inlink, c = crosslink)
        number_links:     number of links
        seed:             fixes the random seed for the link generation
    Returns
        a random sample of number_links links which are of the specified type.
    """
    number_nodes = len(tree.nodes())
    # generate a list of all possible links (all link types)
    links_all = all_links(number_nodes)
    
    # if we include all link types, we immediately return a sample of appropriate size
    types_to_include.sort()
    if types_to_include == ['c', 'i', 'u']:
        fix_seed(seed)
        return sample(links_all, number_links)
    
    # otherwise, we calculate the link types for each link, then filter accordingly and then sample.
    else:
        ncas = find_nca(tree, root, links_all)
        link_types = check_link_types(links_all, ncas, root)
        allowed_links = filter_links_by_type(links_all, link_types, types_to_include)
        fix_seed(seed)
        return sample(allowed_links, number_links)

    
    
def create_directory(filepath):
    """
    creates a directory according to the specified filepath if it does not exist.
    """
    if not exists(filepath):
        makedirs(filepath)

def find_C_e(tree, edge, root):
    """
    Parameters
        tree: tree under consideration
        edge: edge of the tree
        root: root of the tree
    Returns
        list of nodes of the disconnected component - obtained by removing e from the tree - not containing the root.
    """
    # remove given edge from the tree
    tree.remove_edge(edge[0], edge[1])
    # retreive the two disconnected components obtained from removing edge
    component_1, component_2 = list(nx.connected_components(tree))
    # we have to add the edges back to the tree because we actually manipulate the original tree here
    # (we are not working with a copy of the defined tree for computational cost reasons).
    tree.add_edge(edge[0], edge[1])
    if root in component_1:
        return component_2
    else:
        return component_1


def get_instance(n, factor, seed):
    """
    Parameters:
        n:      number of nodes in tree
        factor: defines how many links to generate. We generate factor * (# of edges of tree) links.
        seed:   fixes the random seed for the link generation
    Returns
        a tree T and a list of links links such that len(links) = factor * (# of edges in T) corresponding to the fixed seed.
    """
    T = nx.random_tree(n, seed = seed)
    if factor == 'allLinks':
        number_links = int(n*(n-1)/2)
    else:
        number_links = factor * len(T.edges())
    links = generate_random_links_by_type(T, 1, ['c', 'i', 'u'], number_links, seed = seed + 1)
    return T, links


def get_root(tree):
    """
    Parameters
        tree: tree under consideration
    Returns
        a node of the tree that is not a leaf.
    """
    for potential_root in tree.nodes():
        if tree.degree(potential_root) != 1:
            return potential_root