#### VERSION DESCRIPTION
# Difference to the matching heuristic in version 1:
# instead of choosing one leaf-to-non-leaf whenever there are no leaf-to-leaf links, we pick a random leaf-to-non-leaf link for every leaf.

from random import seed as fix_seed
from random import sample

import networkx as nx
import numpy as np
from tree_augmentation.general import *
from tree_augmentation.natural_lp import *

from tqdm.notebook import tqdm

def find_links_in_leaves(links, leaves):
    """
    Parameters
        links:  list of links
        leaves: list of leaf nodes
    Returns
        list of links both of whose endpoints are leaves
    """
    links_in_leaves = []
    for link in links:
        if link[0] in leaves and link[1] in leaves:
            links_in_leaves.append(link)
    return links_in_leaves


def find_random_link_per_leaf(links, leaves):
    """
    Parameters
        links:  list of links
        leaves: list of leafes of the given tree
    Returns
        list of links obtained as follows: for each leaf v, pick one random link that is adjacent to v.
        If there is at least one leaf v that has no link adjacent to it, return an empty list.
    """
    # for each leaf find the links that are adjacent to the given leaf
    leaf_link_mapping = {}
    for leaf in leaves:
        links_adjacent_to_leaf = [(link[0], link[1]) for link in links if link[0] == leaf or link[1] == leaf]
        # if there are no links that are adjacent to the given leaf, return empty list
        if len(links_adjacent_to_leaf) == 0:
            return []
        else:
            leaf_link_mapping[leaf] = links_adjacent_to_leaf

    # for each link adjacent to some leaf, pick a random element
    random_links = []
    for link_list in leaf_link_mapping.values():
        fix_seed(len(leaves))
        random_link = sample(link_list, 1)[0]
        random_links.append(random_link)
    return random_links 
    
    
def find_components_to_contract(tree, links_to_contract):
    """
    Parameters
        tree:              tree under consideration
        links_to_contract: list of links which we wish to contract
    Returns
        generator object storing the connected components of the graph consisting of
        all edges in tree that are covered by the links in links_to_contract
    """
    # obtain list of all edges that are covered by at least one link in links_to_contract
    edges_to_contract = []
    for link in links_to_contract:
        nodes_of_shortest_path = nx.shortest_path(tree, link[0], link[1])
        for j in range(len(nodes_of_shortest_path) - 1):
            edges_to_contract.append((nodes_of_shortest_path[j], nodes_of_shortest_path[j+1]))
            
    # find connected components of the graph consisting of edges_to_contract
    components_to_contract = nx.connected_components(nx.Graph(edges_to_contract))
    return components_to_contract



def get_residual_tree_and_links(tree, links, links_to_contract):
    """
    Parameters
        tree:              tree in which we wish to perform the contraction
        links:             list of all links
        links_to_contract: list of links which we wish to contract
    Returns
        residual_tree:        tree obtained from contracting all edges that 
                              are covered by at least one link in links_to_contract
        clean_residual_links: list of links in residual_tree 
                              (obtained from links by redefining endpoints to be the contracted node)
        link_correspondence:  dict with key being the old link (i.e. link in tree) 
                              and value being the new link (i.e. link in residual_tree)
    """
    residual_tree = tree.copy()
    
    # consider the graph H consisting of all edges that are covered by at least one link in links_to_contract.
    # the goal is to contract the set of nodes of tree that lie in the same connected component in H.
    # below, we find the connected components
    components_to_contract = find_components_to_contract(tree, links_to_contract)
    node_mapping = {}
    
    # for each connected component, contract the nodes in the connected component
    # the new (contracted) vertex will be denoted by the smallest label of the nodes that are contracted.
    for supernode in components_to_contract:
        nodes = sorted(list(supernode))
        for node in nodes[1:]:
            # store the mapping old node -> new node in the node_mapping dictionary
            node_mapping[node] = nodes[0]
            residual_tree = nx.contracted_nodes(residual_tree, nodes[0], node, self_loops = False)
            
    
    # store the mapping old link -> new link in the link_correspondence dict
    # store all links in the residual graph in residual_links
    residual_links = set()
    link_correspondence = {}
    for old_link in links:
        new_u = node_mapping.get(old_link[0], old_link[0])
        new_v = node_mapping.get(old_link[1], old_link[1])
        if new_u != new_v:
            new_link = create_edge(new_u, new_v)
            residual_links.add(new_link)
            link_correspondence[old_link] = new_link
    
    return residual_tree, list(residual_links), link_correspondence



def matching_heuristic_iteration(tree, leaves, links):
    """
    Parameters
        tree:   tree under consideration
        leafes: list of leaves in tree
        links:  list of weighted links
    Returns
        selected_links:      list of links which are chosen to be a part of the optimal solution 
                             in the iteration of the matching heurisitc.
        residual_tree:       tree obtained from contracting all edges that 
                             are covered by at least one link in links_to_contract
        residual_links:      list of links in residual_tree 
                             (obtained from links by redefining endpoints to be the contracted node)
        link_correspondence: dict with key being the old link (i.e. link in tree) 
                             and value being the new link (i.e. link in residual_tree)
    Notes
        We assume that we can perform a full iteration, i.e. the list of leafes must contain at least one element
        and the links must contain at least one element.
    """
    
    if not leaves:
        raise ValueError('The list leafes must contain at least one element')
    
    if not links:
        raise ValueError('The list links must contain at least one element')
    
    # find links that have both their endpoints in leaves
    links_in_leaves = find_links_in_leaves(links, leaves)
    
    
    if not links_in_leaves:
        # for each leaf pick one random link that is leaving this leaf
        random_links = find_random_link_per_leaf(links, leaves)
        # if there is at least one leaf v which has no link adjacent to it, the find_random_link_per_leaf function returns an empty list.
        # In that case, the instance is infeasible, since the edge in the tree adjacent to v is not covered by any link.
        if random_links == []:
            # set up residual tree that is recognised as infeasible in approximation_via_matchingHeuristic
            residual_tree = nx.Graph()
            residual_tree.add_edge('X', 'Y')
            residual_links = []
            selected_links = []
            link_correspondence = {}
            return selected_links, residual_tree, residual_links, link_correspondence
        else:
            matching = random_links
        
    else:
        # if there are links that have both their endpoints in leaves, compute a max cardinality matching
        auxiliary_graph = nx.Graph(links_in_leaves)
        matching_output = nx.max_weight_matching(auxiliary_graph, maxcardinality=True)
        matching = [create_edge(link[0], link[1]) for link in matching_output]
            
    residual_tree, residual_links, link_correspondence = get_residual_tree_and_links(tree, links, matching)
    selected_links = list(matching)
    return selected_links, residual_tree, residual_links, link_correspondence



def get_link_histories(links, link_correspondences):
    """
    Parameters
        links:                list of links for which we wish to obtain the link histories
        link_correspondences: list of dictionaries where dictionary i stores the link correspondences of iteration i
                              each dict has the old link as key and the new link as value.
    Returns
        np array where the entry (i,j) is the link in the residual graph in iteration j+1 
        which corresponds to the original link i. The first column stores the original links.
        I.e. each row i in link_histories stores all transformations of the original link i.
    """
    link_histories = np.empty((len(links), len(link_correspondences) + 1), dtype = object)
    for i, original_link in enumerate(links):
        link_history = [original_link]
        for j in range(len(link_correspondences)):
            # if a link is not in the current link_correspondence dict, 
            # then it stayed unchanged in the current iteration.
            link_history.append(link_correspondences[j].get(link_history[-1], link_history[-1]))
        link_histories[i,:] = link_history
    return link_histories



def get_links_in_original_form(contracted_links, link_histories):
    """
    Parameters
        contracted_links: list of transformed links (i.e. the links could have contracted nodes as endpoints)
        link_histories:   array storing the transformed links at each iteration 
                          (see get_link_histories for more information)
    Returns
        list of links in original form that correspond to the links in contracted_links
    """
    links_in_original_form = []
    for contracted_link in contracted_links:
        for link_history in link_histories:
            if link_history[-1] == contracted_link:
                links_in_original_form.append(link_history[0])
                break
    return links_in_original_form



def approximation_via_matchingHeuristic(tree, links):
    """
    Parameters
        tree:          tree under consideration
        links:         list of links
    Returns
        approximation to the optimal solution using the matching heuristic which works in the following way:
        1. find all links that have both its endpoints in the leafes of tree
        2. 
            a) if such links exist, compute a maximum cardinality matching in the graph H = (T,L)
               where T is the set of leaves of tree and L is the set of links restricted to T
            b) if no such links exists, pick a random link that has one endpoint in the leafes of tree
            add the selected links to the solution
        3. find all edges that are covered by at least one link that we chose in 2
        4. contract these edges.
        5. 
            a) if the contracted tree has only one node left, terminate the algorithm and return a solution
            b) if the contracted tree has more than one node left and no links, 
               terminate the algorithm and deduce that the instance is infeasible
            c) if the contracted tree has more than one node left and there are still links, go back to step 1
               with the contracted tree.
            
    """
    all_selected_links = []
    
    # perform first iteration
    leaves = [x for x in tree.nodes() if tree.degree(x) == 1]
    all_links = [create_edge(link[0], link[1]) for link in links]
    links = all_links
    selected_links, residual_tree, residual_links, link_correspondence = matching_heuristic_iteration(
        tree, leaves, links
    )
    all_selected_links = selected_links
    link_correspondences = [link_correspondence]
    

    while True:
        # if the residual tree only has one node left, return the current (feasible) solution
        if len(residual_tree.nodes()) == 1:
            link_histories = get_link_histories(all_links, link_correspondences)
            selected_links_orig_form = get_links_in_original_form(all_selected_links, link_histories)
            optimal_solution = [1 if link in selected_links_orig_form else 0 for link in all_links] 
            obj_value = np.sum(optimal_solution)
            output = {
                'status': 'opt',
                'optimal_solution': optimal_solution,
                'optimal_obj_value': obj_value,
                'selected_in_order': all_selected_links,
                'selected_links_orig_form': selected_links_orig_form
            }
            return output
        # if the residual tree has one or more edges and no links, the instance is infeasible
        elif len(residual_links) == 0 and len(residual_tree.edges()) >= 1:
            output = {
                'status': 'inf',
                'optimal_solution': [],
                'optimal_obj_value': None
            }
            return output
        # otherwise, perform another iteration of the matching heuristic.
        else:
            tree = residual_tree
            links = residual_links
            leaves = [x for x in tree.nodes() if tree.degree(x) == 1]
            selected_links, residual_tree, residual_links, link_correspondence = matching_heuristic_iteration(
                tree, leaves, links
            )
            all_selected_links += selected_links
            link_correspondences.append(link_correspondence)
            
            
            
            
def run_simulations_approximation_via_matchingHeuristic(
    samples, n, factor, output_directory, disable_progress_bar = False
):
    """
    Parameters
        samples:              number of TAP instances for which to run the algorithms
        n:                    number of nodes of the tree
        factor:               defines how many links to generate. We generate factor * (# of edges of tree) links.
        disable_progress_bar: True or False, whether or not to disable the progress bar
    Returns
        approx_objectives: array of the optimal objective values of the approximation algorithm via matching heuristic
        approx_solutions:  array of the optimal solutions of the approximation algorithm via matching heuristic
        exact_objectives:  array of the optimal objective values of the exact algorithm
        exact_solutions:   array of the optimal solutions of the exact algorithm
        We also save the corresponding arrays in the output_directory folder.
    """
    approx_objectives = []
    approx_solutions = []
    exact_objectives = []
    exact_solutions = []
    for k in tqdm(range(samples), disable = disable_progress_bar, desc = f'MAT2: n = {n}, factor = {factor}'):
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
            approximation_output = approximation_via_matchingHeuristic(tree = T, links = links)
            approx_objectives.append(approximation_output['optimal_obj_value'])
            approx_solutions.append(approximation_output['optimal_solution'])
            
    # save results in the output_directory file
    approx_objectives = np.array(approx_objectives, dtype = object)
    approx_solutions = np.array(approx_solutions, dtype = object)
    exact_objectives = np.array(exact_objectives, dtype = object)
    exact_solutions = np.array(exact_solutions, dtype = object)
    output_filepath = f'{output_directory}/matchingHeuristicBasic_approximation/samples{samples}/n{n}/'
    create_directory(output_filepath)
    filename_prefix = f'samples{samples}_n{n}_factor{factor}_'
    np.save(f'{output_filepath}{filename_prefix}approx_objectives.npy', approx_objectives)
    np.save(f'{output_filepath}{filename_prefix}approx_solutions.npy', approx_solutions)
    np.save(f'{output_filepath}{filename_prefix}exact_objectives.npy', exact_objectives)
    np.save(f'{output_filepath}{filename_prefix}exact_solutions.npy', exact_solutions)
    return approx_objectives, approx_solutions, exact_objectives, exact_solutions


def get_approximation_ratios_of_matchingHeuristic_approximation(samples, n, factor, output_directory):
    """
    Parameters
        samples: number of TAP instances for which to compute the approximation ratio
        n:       number of nodes of the tree in the TAP instance
        factor:  defines how many links to generate. We generate factor * (# of edges of tree) links.
    Returns
        array storing the approximation ratios of the matching heuristic approximation algorithm 
        for each of the (samples) TAP instances
    """
    file_path = f'{output_directory}/matchingHeuristicBasic_approximation/samples{samples}/n{n}/'
    filename_prefix = f'samples{samples}_n{n}_factor{factor}_'
    fp_eo = f'{file_path}{filename_prefix}exact_objectives.npy'
    fp_ao = f'{file_path}{filename_prefix}approx_objectives.npy'
    
    # if the TAP instances have not yet been computed, compute them
    if (not exists(fp_eo)) or (not exists(fp_ao)):
        approx_objectives,_,exact_objectives,_ = run_simulations_approximation_via_matchingHeuristic(
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