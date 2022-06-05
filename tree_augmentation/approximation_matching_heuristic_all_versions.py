#### VERSION DESCRIPTION
# Difference to the matching heuristic in version 6:
# Whenever there does not exist a leaf-to-leaf link, we proceed with yet another matching approach:
# We select a set S of links that connect a leaf to a vertex with maximum leaf-distance (max_leaf_distance), 
# leaf-distance(v) = distance from v to the closest leaf.
# If no such link exists, pick all links that go from a leaf to a node with leaf-distance max_leaf_distance -1. 
# Proceed with that iteratively.
# Once we have found such a set S, we compute a matching in this set S.

from random import seed as fix_seed
from random import sample

import networkx as nx
import numpy as np
from tree_augmentation.general import *
from tree_augmentation.natural_lp import *

from tqdm.notebook import tqdm


def find_bad_links(tree, links, root, output_certificate = False):
    """
    Parameters
        tree:               tree of the TAP instance
        links:              links of the TAP instance
        root:               root of the tree
        output_certificate: True or False. If True, output the edge(s) that certify that the outputted links are bad. 
                            If False, only return set of bad links
    Returns
        subset of links that are "bad". A link l = {u,v} is bad if there exists an edge e for which all links covering e
        have one endpoint being u or v. Furthermore, u and v both have to lie in the set C_e (which is the set 
        of nodes of the disconnected component - obtained by removing e from the tree - not containing the root).
    Notes
        The idea of this algorithm is to loop over the edges e and find all links that cover e. We then retreive the endpoints
        of all links covering e that lie inside the set C_e. We call this set T_e. If |T_e| = 2, then we have a potential bad link.
        We now only need to check whether there exists a link l with l = T_e.
    """
    tree_edges = list(tree.edges())
    number_links = len(links)
    number_edges = len(tree_edges)
    edge_link_cover_matrix = create_edge_link_cover_matrix(tree, links, tree_edges, number_edges, number_links)
    
    # store links in a numpy array
    links_np = np.empty(len(links), dtype = object)
    links_np[:] = links

    bad_links = set()
    bad_links_certificates = []
    for k, e in enumerate(tree_edges):
        # obtain list of links that cover e
        links_covering_e = links_np[edge_link_cover_matrix[k,:] == 1]
        
        if len(links_covering_e) <2:
            continue
        else:
            # T_e is the set of nodes that lie in C_e and are an endpoint of the links covering e
            T_e = set()
            C_e = find_C_e(tree, e, root)
            
            # this variable will be set to True whenever we found a link that shows that e doesn't allow for a bad link
            # (i.e. if we found a link that would cause T_e to have cardinality greater than 2)
            e_doesnt_allow_for_badLink = False
            
            for link in links_covering_e:
                if link[0] in C_e:
                    T_e.add(link[0])
                elif link[1] in C_e:
                    T_e.add(link[1])
                if len(T_e) > 2:
                    # if there are more than two nodes that lie in C_e, are leafes and are an endpoint of at least one
                    # leaf that covers e, e does not allow for a bad link
                    e_doesnt_allow_for_badLink = True
                    break
                    
            if e_doesnt_allow_for_badLink:
                continue
            else:
                # if T_e contains only two nodes, it qualifies for being a bad link. 
                # but only if the two nodes correspond to a link, we add the link to the bad link list.
                if len(T_e) == 2:
                    potential_bad_link = create_edge(list(T_e)[0], list(T_e)[1])
                    if potential_bad_link in links:
                        bad_links.add(potential_bad_link)
                        bad_links_certificates.append(e)
    if output_certificate:
        return bad_links, bad_links_certificates
    else:
        return bad_links
    
    
    
def find_shadows(tree, links):
    """
    Parameters
        tree:  tree of the TAP instance
        links: links in the TAP instance
    Returns
        set of shadows
    Notes
        a link is called a shadow if it covers a set of edges of tree that is also covered by some other link.
    """
    shadows = set()
    links_without_shadows = set(links)

    # for each link, find the shortest path between its two endpoints. 
    # If there is a link that has both its endpoints on the path, 
    # add it to the shadows set and remove it from the links_without_shadows set
    for link in links:
        P = nx.shortest_path(tree, link[0], link[1])
        detected_shadows = set()
        for potential_shadow in links_without_shadows:
            if potential_shadow != link and potential_shadow[0] in P and potential_shadow[1] in P:
                detected_shadows.add(potential_shadow)
        shadows = shadows.union(detected_shadows)
        links_without_shadows = links_without_shadows.difference(detected_shadows)
        
    return shadows







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

def find_leaf_distances(tree):
    """
    Parameters
        tree: tree under consideration
    Returns
        dict with nodes of tree as keys. The value of each node v is the distance to the closest leaf. 
        Leaf nodes have value 0.
    Notes
        the algorithm assigns value 0 to all leaf nodes and value 1 to all nodes that are neighbours of leaf nodes.
        We then delete all leaves from the tree and look at all nodes neighbouring the leaves in the newly obtained tree.
        These nodes will get value 2. We proceed in a similar fashion.
    """
    leaf_distances = {}
    leaves = [x for x in tree.nodes() if tree.degree(x) <= 1]
    for leaf in leaves:
        leaf_distances[leaf] = 0
    
    i = 1
    # iterate until we have assigned a value to each node in the tree
    while len(tree.nodes()) > 0:
        # find leaves in the tree
        leaves = [x for x in tree.nodes() if tree.degree(x) <= 1]
        leaf_neighbours = set()
        # find all nodes neighbouring the leaves of the current tree 
        for leaf in leaves:
            leaf_neighbours = leaf_neighbours.union(set(tree.neighbors(leaf)))
        # find all nodes in tree that have already been assigned a distance, find their neighbours and add them to the leaf_neighbours set
        for already_processed_node in leaf_distances.keys():
            if already_processed_node in tree.nodes():
                leaf_neighbours = leaf_neighbours.union(set(tree.neighbors(already_processed_node)))
        # if a node in leaf_neighbours already has a value, then this value is strictly smaller than i
        # since we are looking for the smallest distance to a leaf, we do not re-assign in that case.
        for leaf_neighbour in leaf_neighbours:
            if leaf_neighbour not in leaf_distances:
                leaf_distances[leaf_neighbour] = i
        # remove all leaves from the tree
        tree.remove_nodes_from(leaves)
        i += 1
    return leaf_distances

def find_links_for_matching(leaf_distances, links):
    """
    Parameters
        leaf_distances: dict storing the distance of every node to the closest leaf
        links:          list of links under consideration
    Returns
        list of links that should be considered for the matching algorithm.
    Notes
        The function assumes that there are no leaf-to-leaf links. 
        We then try all links that connect a leaf and a vertex with maximum leaf-distance max_leaf_distance, 
        where leaf-distance is the distance to the closest leaf.
        If no such link exists, pick all links that go from a leaf to a node with leaf distance max_leaf_distance -1.
        Proceed with that iteratively.
    """
    # find maximum leaf-distance
    max_leaf_distance = max(leaf_distances.values())
    for current_leaf_distance in reversed(range(1, max_leaf_distance + 1)):
        links_for_matching = []
        # find all links (u,v) such that u (or v) is a leaf and v (or u) has leaf distance = current_leaf_distance
        for u,v in links:
            if leaf_distances[u] == 0 and leaf_distances[v] == current_leaf_distance:
                links_for_matching.append((u,v))
            elif leaf_distances[v] == 0 and leaf_distances[u] == current_leaf_distance:
                links_for_matching.append((u,v))
        # if there are links (u,v) such that u (or v) is a leaf and v (or u) has leaf distance = current_leaf_distance, 
        # return them. Otherwise, reduce current_leaf_distance by 1 and try again.
        if links_for_matching:
            return links_for_matching


def find_max_weight_matching(links, tree):
    """
    Parameters
        links: list of links among which to compute the matching
        tree:  tree of the TAP instance
    Returns
        a maximum cardinality matching in the graph which has edges links.
    Notes
        The function finds a matching that covers a lot of edges. For that, let C_l be the number of edges that link l covers.
        And define C = sum_l C_l. We now define the weight for each link l to be C + C_l. Note that by picking the weights
        in this way, we always calculate a maximum cardinality matching. 
        The only issue with this approach: we count edges several times (if the edge is covered by several links). 
    """
    # count how many edges each link covers and store result in a dictionary
    cover_matrix = create_edge_link_cover_matrix(tree, links, list(tree.edges()), len(tree.edges()), len(links))
    number_edges_covered_by_links = np.sum(cover_matrix, axis = 0)
    link_weights = np.sum(cover_matrix) + number_edges_covered_by_links
    d = {}
    for i in range(len(links)):
        d[links[i]] = link_weights[i]
    # set up the graph in which to calculate the maximum weight matching
    auxiliary_graph = nx.Graph(links)
    nx.set_edge_attributes(auxiliary_graph, d, 'weight')
    
    # compute and output the maximum weight matching
    matching_output = nx.max_weight_matching(auxiliary_graph, maxcardinality=True, weight = 'weight')
    matching = [create_edge(link[0], link[1]) for link in matching_output]
    return matching



def get_matching_with_number_covered_edges(links_in_leaves, cover_matrix):
    """
    Parameters
        links_in_leaves: list of links both of whose endpoints are leaves
        cover_matrix:    edge-link cover matrix for a TAP instance on some tree with links_in_leaves as links
    Returns
        a maximum caridnality matching in the graph consisting of the links_in_leaves edges, 
        and the number of edges this matching covers in the tree of the TAP instance 
        (the tree corresponding to the cover_matrix)
    """
    # calculate the maximum cardinality matching
    auxiliary_graph = nx.Graph(links_in_leaves)
    matching_output = nx.max_weight_matching(auxiliary_graph, maxcardinality=True)
    matching = [create_edge(link[0], link[1]) for link in matching_output]
    
    # convert the lists links_in_leaves and matching to numpy arrays
    links_in_leaves_np = np.empty(len(links_in_leaves), dtype = object)
    links_in_leaves_np[:] = links_in_leaves
    matching_np = np.empty(len(matching), dtype = object)
    matching_np[:] = matching
    
    # retreive the rows of the link-edge cover matrix (transpose of cover_matrix) that correspond to the links in the matching
    cover_matrix_matching = np.transpose(cover_matrix)[np.isin(links_in_leaves_np, matching_np)]
    # calculate the sum over the columns and count how many sums are greater than 0. 
    # if a sum is greater than 0, then the corresponding edge is covered by at least one link
    number_covered_edges_by_matching = np.sum(np.sum(cover_matrix_matching, axis = 0) > 0)
    return matching, number_covered_edges_by_matching


def find_locally_best_matching(links_in_leaves, tree):
    """
    Parameters
        links_in_leaves: list of links both of whose endpoints are leaves
        tree:            tree of the TAP instance
    Returns
        a maximum cardinality matching in the graph which has edges links_in_leaves.
    Notes
        The function finds a ``locally best`` matching, i.e. if we find a matching M which does not cover all edges in tree
        then we remove remove one link of M from links_in_leaves at a time, re-compute the matching and add the link back to 
        links_in_leaves. We do this for every link in the initially chosen matching. We store the number of edges covered
        by each matching and return the best matching.
    """
    # if there is only one link, then this link is the unique maximum cardinality matching
    if len(links_in_leaves) == 1:
        return links_in_leaves
    
    
    number_tree_edges = len(tree.edges())
    cover_matrix = create_edge_link_cover_matrix(
        tree, links_in_leaves, list(tree.edges()), number_tree_edges, len(links_in_leaves)
    )
    matching, number_covered_edges_by_matching = get_matching_with_number_covered_edges(links_in_leaves, cover_matrix)
    
    # if all links are included in the matching or if all edges are covered by the given matching, 
    # then there is no way of improving the matching
    if len(links_in_leaves) == len(matching) or number_covered_edges_by_matching == number_tree_edges:
        return matching
    
    # otherwise, we try to find a better matching as described in Notes (see above)
    
    matchings = [matching]
    number_edges_covered = [number_covered_edges_by_matching]
    
    for matching_link in matching:
        # delete matching_edge from links_in_leaves
        index_of_matching_link = links_in_leaves.index(matching_link)
        links_in_leaves.pop(index_of_matching_link)

        # define the remove the column in cover_matrix corresponding to the deleted link (i.e. matching_edge)
        reduced_cover_matrix = np.delete(cover_matrix, index_of_matching_link, axis = 1)

        matching, number_covered_edges_by_matching = get_matching_with_number_covered_edges(
            links_in_leaves, reduced_cover_matrix
        )
        
        # if the current matching covers all edges in the tree, then it is the optimal matching
        if number_covered_edges_by_matching == number_tree_edges:
            return matching
        
        # if the current matching consists of less links than the original matching, then it is not a maximum cardinality matching.
        if len(matching) == len(matchings[0]):
            matchings.append(matching)
            number_edges_covered.append(number_covered_edges_by_matching)
        elif len(matching) > len(matchings[0]):
            raise ValueError('The initially computed matching is not of maximum cardinality')
        

        # insert the matching 
        links_in_leaves.insert(index_of_matching_link, matching_link)
        
    best_matching_index = number_edges_covered.index(max(number_edges_covered))
    best_matching = matchings[best_matching_index]
    return best_matching


def find_best_links_per_leaf(links, leaves, tree):
    """
    Parameters
        links:  list of links
        leaves: list of leafes of the given tree
        tree:   tree from the TAP instance
    Returns
        list of links obtained as follows: for each leaf v, pick the link that is incident to v which icreases the number 
        of edges covered the most. If there is at least one leaf v that has no link incident to it, return an empty list.
    Notes
        For the first leaf v, we choose the link l_1 that covers the most edges among all links that are incident to v.
        For the second leaf w, suppose the links l_{2,1},...,l_{2,k} are incident to w. We choose the one link l_{2,j}
        such that l_1 and l_{2,j} together cover the most edges.
        Proceed in this fashion for all subsequent leaves.
    """
    # for each leaf find the links that are incident to the given leaf
    leaf_link_mapping = {}
    for leaf in leaves:
        links_incident_to_leaf = [(link[0], link[1]) for link in links if link[0] == leaf or link[1] == leaf]
        # if there are no links that are incident to the given leaf, return empty list
        if len(links_incident_to_leaf) == 0:
            return []
        else:
            leaf_link_mapping[leaf] = links_incident_to_leaf
            
    # for each link, store the set of edges it covers
    link_edge_cover = {}
    for link in links:
        P_l_nodeset = nx.shortest_path(tree, link[0], link[1])
        P_l_edgeset = set()
        for i in range(1, len(P_l_nodeset)):
            P_l_edgeset.add(create_edge(P_l_nodeset[i-1], P_l_nodeset[i]))
        link_edge_cover[link] = P_l_edgeset
        
    edges_covered_all = set()
    best_links = list()
    # loop over leaves
    for link_list in leaf_link_mapping.values():
        edgeset_covered_if_link_added = dict()
        number_edges_covered_if_link_added = dict()
        # loop over links incident to the current leaf
        for link in link_list:
            # store the set (and its size) of edges that are covered by the current link and all previously selected links
            edges_covered_by_link = link_edge_cover[link]
            edgeset = edges_covered_all.union(edges_covered_by_link)
            edgeset_covered_if_link_added[link] = edgeset
            number_edges_covered_if_link_added[link] = len(edgeset)
        # select the link that covers the most edges together with all previously selected links
        best_link = max(number_edges_covered_if_link_added, key = number_edges_covered_if_link_added.get)
        best_links.append(best_link)
        # update the set of covered links
        edges_covered_all = edgeset_covered_if_link_added[best_link].copy()
    # return all selected links
    return best_links




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



def matching_heuristic_iteration(tree, leaves, links, matching_procedure, leaf_nonleaf_selection):
    """
    Parameters
        tree:               tree under consideration
        leafes:             list of leaves in tree
        links:              list of weighted links
        matching_procedure:     'max_weight' or 'iterative_removal'. Defines the algorithm to use whenever we compute a matching 
                                if 'max_weight', we follow a weighted matching approach. See notes in find_max_weight_matching for details.
                                if 'iterative_removal', we iteratively remove one edge from the initially computed maximum cardinality
                                matching and re-compute the matching. Out of all matchings found, pick the one with most number of edges
                                covered. See notes in find_locally_best_matching for details.
        leaf_nonleaf_selection: 'greedy' or 'leaf_distance_matching'. Defines the algorithm to use whenever the given TAP instance has no
                                leaf-to-leaf links. 
                                if 'greedy', we use a greedy algorithm that aims to maximise the number of edges covered. 
                                See notes in find_best_links_per_leaf for deetails.
                                if 'leaf_distance_matching',  find all links that connect a leaf and a vertex with maximal leaf-distance 
                                and compute a matching on these links. See notes in find_links_for_matching for details.
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
        raise ValueError('The list leaves must contain at least one element')
    
    if not links:
        raise ValueError('The list links must contain at least one element')
    
    
    root = get_root(tree)
    shadows = find_shadows(tree, links)
    no_shadow_links = list(set(links) - shadows)
    
    bad_links = find_bad_links(tree, no_shadow_links, root)
    good_links = list(set(no_shadow_links) - bad_links)
    
    # find links that have both their endpoints in leaves
    links_in_leaves = find_links_in_leaves(good_links, leaves)
    
    if not links_in_leaves:
        if leaf_nonleaf_selection == 'greedy':
            # for each leaf pick one link that is leaving this leaf (using a greedy algorithm that maximises the number of edges covered)
            selected_links = find_best_links_per_leaf(good_links, leaves, tree)
            # if there is at least one leaf v which has no link incident to it, the find_best_links_per_leaf function returns an empty list.
            # In that case, the instance is infeasible, since the edge in the tree incident to v is not covered by any link.
            if selected_links == []:
                # set up residual tree that is recognised as infeasible in approximation_via_matchingHeuristic
                residual_tree = nx.Graph()
                residual_tree.add_edge('X', 'Y')
                residual_links = []
                selected_links = []
                link_correspondence = {}
                return selected_links, residual_tree, residual_links, link_correspondence
            else:
                matching = selected_links
        elif leaf_nonleaf_selection == 'leaf_distance_matching':
            leaf_distances = find_leaf_distances(tree.copy())
            links_for_matching = find_links_for_matching(leaf_distances, good_links)
            if matching_procedure == 'max_weight':
                matching = find_max_weight_matching(links_for_matching, tree)
            elif matching_procedure == 'iterative_removal':
                matching = find_locally_best_matching(links_for_matching, tree)
            else:
                raise ValueError(f"The matching_procedure must not be {matching_procedure}. Use 'max_weight' or 'iterative_removal'.")
        else:
            raise ValueError(f"The leaf_nonleaf_selection must not be {leaf_nonleaf_selection}. Use 'greedy' or 'leaf_distance_matching'.")
    else:
        if matching_procedure == 'max_weight':
            # if there are links that have both their endpoints in leaves, compute a max cardinality matching with maximum weight
            # see comments in the function definition for further details
            matching = find_max_weight_matching(links_in_leaves, tree)
        elif matching_procedure == 'iterative_removal':
            # if there are links that have both their endpoints in leaves, compute the locally best max cardinality matching
            # see comments in the function definition for further details
            matching = find_locally_best_matching(links_in_leaves, tree)
        else:
            raise ValueError(f"The matching_procedure must not be {matching_procedure}. Use 'max_weight' or 'iterative_removal'.")
            
    residual_tree, residual_links, link_correspondence = get_residual_tree_and_links(tree, good_links, matching)
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
    Notes
        We later iterate over the result from top to bottom. Since we wish to select the links that correspond to good links in the firtst
        iteration, we simply store the histories fo the good links on top.
    """
    link_histories = np.empty((len(links), len(link_correspondences) + 1), dtype = object)
    good_links = link_correspondences[0].keys()
    bad_links = set(links) - set(good_links)
    
    for i, good_link in enumerate(good_links):
        link_history = [good_link]
        for j in range(len(link_correspondences)):
            # if a link is not in the current link_correspondence dict, 
            # then it stayed unchanged in the current iteration.
            link_history.append(link_correspondences[j].get(link_history[-1], link_history[-1]))
        link_histories[i,:] = link_history
    
    for i, bad_link in enumerate(bad_links):
        link_history = [bad_link]
        for j in range(len(link_correspondences)):
            # if a link is not in the current link_correspondence dict, 
            # then it stayed unchanged in the current iteration.
            link_history.append(link_correspondences[j].get(link_history[-1], link_history[-1]))
        link_histories[i+len(good_links),:] = link_history
        
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



def approximation_via_matchingHeuristic(tree, links, matching_procedure, leaf_nonleaf_selection):
    """
    Parameters
        tree:                   tree under consideration
        links:                  list of links
        matching_procedure:     'max_weight' or 'iterative_removal'. Defines the algorithm to use whenever we compute a matching 
                                if 'max_weight', we follow a weighted matching approach. See notes in find_max_weight_matching for details.
                                if 'iterative_removal', we iteratively remove one edge from the initially computed maximum cardinality
                                matching and re-compute the matching. Out of all matchings found, pick the one with most number of edges
                                covered. See notes in find_locally_best_matching for details.
        leaf_nonleaf_selection: 'greedy' or 'leaf_distance_matching'. Defines the algorithm to use whenever the given TAP instance has no
                                leaf-to-leaf links. 
                                if 'greedy', we use a greedy algorithm that aims to maximise the number of edges covered. 
                                See notes in find_best_links_per_leaf for deetails.
                                if 'leaf_distance_matching',  find all links that connect a leaf and a vertex with maximal leaf-distance 
                                and compute a matching on these links. See notes in find_links_for_matching for details.
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
        tree, leaves, links, matching_procedure, leaf_nonleaf_selection
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
                tree, leaves, links, matching_procedure, leaf_nonleaf_selection
            )
            all_selected_links += selected_links
            link_correspondences.append(link_correspondence)
            
            
            
            
def run_simulations_approximation_via_matchingHeuristic(
    matching_procedure, leaf_nonleaf_selection, samples, n, factor, output_directory, disable_progress_bar = False
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
    for k in tqdm(
        range(samples), 
        disable = disable_progress_bar, 
        desc = f'{matching_procedure}, {leaf_nonleaf_selection}: n = {n}, factor = {factor}',
        leave = False
    ):
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
            
            # compute the approximate solution using the matching approximation algorithm
            approximation_output = approximation_via_matchingHeuristic(T, links, matching_procedure, leaf_nonleaf_selection)
            approx_objectives.append(approximation_output['optimal_obj_value'])
            approx_solutions.append(approximation_output['optimal_solution'])
            
    # save results in the output_directory file
    approx_objectives = np.array(approx_objectives, dtype = object)
    approx_solutions = np.array(approx_solutions, dtype = object)
    exact_objectives = np.array(exact_objectives, dtype = object)
    exact_solutions = np.array(exact_solutions, dtype = object)
    output_filepath = f'./{output_directory}/matchingHeuristic_approximation/matching_{matching_procedure}/leafNonLeaf_{leaf_nonleaf_selection}/samples{samples}/n{n}/'
    create_directory(output_filepath)
    filename_prefix = f'samples{samples}_n{n}_factor{factor}_'
    np.save(f'{output_filepath}{filename_prefix}approx_objectives.npy', approx_objectives)
    np.save(f'{output_filepath}{filename_prefix}approx_solutions.npy', approx_solutions)
    np.save(f'{output_filepath}{filename_prefix}exact_objectives.npy', exact_objectives)
    np.save(f'{output_filepath}{filename_prefix}exact_solutions.npy', exact_solutions)
    return approx_objectives, approx_solutions, exact_objectives, exact_solutions


def get_approximation_ratios_of_matchingHeuristic_approximation(
    matching_procedure, leaf_nonleaf_selection, samples, n, factor, output_directory
):
    """
    Parameters
        samples: number of TAP instances for which to compute the approximation ratio
        n:       number of nodes of the tree in the TAP instance
        factor:  defines how many links to generate. We generate factor * (# of edges of tree) links.
    Returns
        array storing the approximation ratios of the matching heuristic approximation algorithm 
        for each of the (samples) TAP instances
    """
    file_path = f'{output_directory}/matchingHeuristic_approximation/matching_{matching_procedure}/leafNonLeaf_{leaf_nonleaf_selection}/samples{samples}/n{n}/'
    filename_prefix = f'samples{samples}_n{n}_factor{factor}_'
    fp_eo = f'{file_path}{filename_prefix}exact_objectives.npy'
    fp_ao = f'{file_path}{filename_prefix}approx_objectives.npy'
    
    # if the TAP instances have not yet been computed, compute them
    if (not exists(fp_eo)) or (not exists(fp_ao)):
        approx_objectives,_,exact_objectives,_ = run_simulations_approximation_via_matchingHeuristic(
            matching_procedure, leaf_nonleaf_selection, samples, n, factor, output_directory
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