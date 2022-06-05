import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tree_augmentation.general import *
from tree_augmentation.natural_lp import *
import tree_augmentation.approximation_uplinks as upl
import tree_augmentation.approximation_naiveRounding as nrd
import tree_augmentation.approximation_iterativeRounding as ird
import tree_augmentation.approximation_matching_heuristic_all_versions as mat
import tree_augmentation.approximation_matching_heuristic_basic as mat_b



ALL_ALGORITHMS = ['NLP', 'EXT', 'UPL', 'NRD', 'IRD', 'MAT_B', 'MAT_WG', 'MAT_WM', 'MAT_RG', 'MAT_RM']
OUTPUT_FOLDER = 'saved_outputs'



def compare_outputs(T, links, root, algorithms_to_compare = ALL_ALGORITHMS):
    """
    Parameters
        T:                     tree of the TAP instance for which to compare the algorithms
        links:                 links of the TAP instance for which to compare the algorithms
        root:                  root of T (needed for drawing and the algorithms UPL, MBL)
        algorithms_to_compare: list of algorithm names (abbreviations) which we wish to compare where
                               NLP   : natural LP relaxation
                               EXT   : exact solution (via integer programming)
                               UPL   : approximation via uplinks
                               NRD   : approximation via naive rounding
                               IRD   : approximation via iterative rounding
                               MAT_B : approximation via matching heuristic, basic version: 
                                       any max cardinality matching, random leaf-non-leaf selection
                               MAT_WG: approximation via matching heuristic with 
                                       matching procedure max weight (W) and leaf-non-leaf selection via greedy algorithm (G)
                               MAT_WM: approximation via matching heuristic with
                                       matching procedure maw weight (W) and leaf-non-leaf selection via leaf distance matching (M)
                               MAT_RG: approximation via matching heuristic with
                                       matching procedure iterative removal (R) and leaf-non-leaf selection via greedy algorithm (G)
                               MAT_RM: approximation via matching heuristic with
                                       matching procedure iterative removal (R) and leaf-non-leaf selection via leaf distance matching (M)
    Returns
        pandas dataframe storing the optimal value returned by the selected algorithms for the given instance and the links chosen.
    """
    draw_tree(T, root, links)
    data_all = []

    # get outputs of natural LP relaxation
    if 'NLP' in algorithms_to_compare:
        sol_NLP = solve_naturalLP(T, links, link_weights=None, variable_type=GRB.CONTINUOUS)
        data_NLP = {'algorithm': 'natural LP relaxation', 'value': sol_NLP['optimal_obj_value']}
        for i in range(len(links)):
            data_NLP[links[i]] = sol_NLP['optimal_solution'][i]
        data_all.append(data_NLP)
        
    # get outputs of exact solution
    if 'EXT' in algorithms_to_compare:
        sol_EXT = solve_naturalLP(T, links, link_weights=None, variable_type=GRB.BINARY)
        data_EXT = {'algorithm': 'exact', 'value': sol_EXT['optimal_obj_value']}
        for i in range(len(links)):
            data_EXT[links[i]] = sol_EXT['optimal_solution'][i]
        data_all.append(data_EXT)

    # get outputs for approximation via uplinks
    if 'UPL' in algorithms_to_compare:
        sol_UPL = upl.approximation_via_uplinks(T, root, links)
        data_UPL = {'algorithm': 'uplinks', 'value': sol_UPL['optimal_obj_value']}
        for i in range(len(links)):
            data_UPL[links[i]] = sol_UPL['optimal_solution'][i]
        data_all.append(data_UPL)

    # get outputs for approximation via naive rounding
    if 'NRD' in algorithms_to_compare:
        sol_NRD = nrd.approximation_via_naiveRounding(T, links, link_weights = None)
        data_NRD = {'algorithm': 'naive rounding', 'value': sol_NRD['optimal_obj_value']}
        for i in range(len(links)):
            data_NRD[links[i]] = sol_NRD['optimal_solution'][i]
        data_all.append(data_NRD)
        
    # get outputs for approximation via iterative rounding
    if 'IRD' in algorithms_to_compare:
        sol_IRD = ird.approximation_via_iterativeRounding(T, links)
        data_IRD = {'algorithm': 'iterative rounding', 'value': sol_IRD['optimal_obj_value']}
        for i in range(len(links)):
            data_IRD[links[i]] = sol_IRD['optimal_solution'][i]
        data_all.append(data_IRD)

    # get outputs for matching heuristic: matching procedure max_weight, leaf-non-leaf selection greedy
    if 'MAT_B' in algorithms_to_compare:
        sol_MAT = mat_b.approximation_via_matchingHeuristic(T, links)
        data_MAT = {'algorithm': 'matching heuristic B', 'value': sol_MAT['optimal_obj_value']}
        for i in range(len(links)):
            data_MAT[links[i]] = sol_MAT['optimal_solution'][i]
        data_all.append(data_MAT)
    
    # get outputs for matching heuristic: matching procedure max_weight, leaf-non-leaf selection greedy
    if 'MAT_WG' in algorithms_to_compare:
        sol_MAT = mat.approximation_via_matchingHeuristic(T, links, 'max_weight', 'greedy')
        data_MAT = {'algorithm': 'matching heuristic WG', 'value': sol_MAT['optimal_obj_value']}
        for i in range(len(links)):
            data_MAT[links[i]] = sol_MAT['optimal_solution'][i]
        data_all.append(data_MAT)
        
    # get outputs for matching heuristic: matching procedure max_weight, leaf-non-leaf selection leaf distance matching
    if 'MAT_WM' in algorithms_to_compare:
        sol_MAT = mat.approximation_via_matchingHeuristic(T, links, 'max_weight', 'leaf_distance_matching')
        data_MAT = {'algorithm': 'matching heuristic WM', 'value': sol_MAT['optimal_obj_value']}
        for i in range(len(links)):
            data_MAT[links[i]] = sol_MAT['optimal_solution'][i]
        data_all.append(data_MAT)
        
    # get outputs for matching heuristic: matching procedure iterative removal, leaf-non-leaf selection greedy
    if 'MAT_RG' in algorithms_to_compare:
        sol_MAT = mat.approximation_via_matchingHeuristic(T, links, 'iterative_removal', 'greedy')
        data_MAT = {'algorithm': 'matching heuristic RG', 'value': sol_MAT['optimal_obj_value']}
        for i in range(len(links)):
            data_MAT[links[i]] = sol_MAT['optimal_solution'][i]
        data_all.append(data_MAT)
        
    # get outputs for matching heuristic: matching procedure iterative removal, leaf-non-leaf selection leaf distance matching
    if 'MAT_RM' in algorithms_to_compare:
        sol_MAT = mat.approximation_via_matchingHeuristic(T, links, 'iterative_removal', 'leaf_distance_matching')
        data_MAT = {'algorithm': 'matching heuristic RM', 'value': sol_MAT['optimal_obj_value']}
        for i in range(len(links)):
            data_MAT[links[i]] = sol_MAT['optimal_solution'][i]
        data_all.append(data_MAT)
        
    
        
    df = pd.DataFrame(data_all)
    df.replace(0.0, '', inplace = True)
    return df





def solve_instance(n, factor, seed, selected_only = True, algorithms_to_compare = ALL_ALGORITHMS):
    """
    Parameters:
        n:                     number of nodes in tree
        factor:                defines how many links to generate. We generate factor * (# of edges of tree) links.
        seed:                  fixes the random seed for the link and tree generation
        selected_only:         True or False. 
                               If True, remove all columns corresponding to links not chosen by any algorithm.  If False, leave all columns.
        algorithms_to_compare: list of algorithm names (abbreviations) which we wish to compare where
                               NLP   : natural LP relaxation
                               EXT   : exact solution (via integer programming)
                               UPL   : approximation via uplinks
                               NRD   : approximation via naive rounding
                               IRD   : approximation via iterative rounding
                               MAT_B : approximation via matching heuristic, basic version: 
                                       any max cardinality matching, random leaf-non-leaf selection
                               MAT_WG: approximation via matching heuristic with 
                                       matching procedure max weight (W) and leaf-non-leaf selection via greedy algorithm (G)
                               MAT_WM: approximation via matching heuristic with
                                       matching procedure maw weight (W) and leaf-non-leaf selection via leaf distance matching (M)
                               MAT_RG: approximation via matching heuristic with
                                       matching procedure iterative removal (R) and leaf-non-leaf selection via greedy algorithm (G)
                               MAT_RM: approximation via matching heuristic with
                                       matching procedure iterative removal (R) and leaf-non-leaf selection via leaf distance matching (M)
    Returns:
        plots the instance and prints a table storing the objective values and selected links by the corresponding algorithm
    """
    T, links = get_instance(n, factor, seed)
    
    for potential_root in T.nodes():
        if T.degree(potential_root) != 1:
            break
    output_table = compare_outputs(T, links, potential_root, algorithms_to_compare = algorithms_to_compare)
    output_table.replace('', np.nan, inplace=True)
    output_table.dropna(how = 'all', axis = 1, inplace = True)
    output_table.replace(np.nan, '', inplace = True)
    return output_table


def plot_joint_hist(n, factor, axis, to_include = ALL_ALGORITHMS[2:]):
    """
    Parameters:
        n:          number of nodes in tree
        factor:     defines how many links to generate. We generate factor * (# of edges of tree) links.
        axis:       axis on which to plot the historgram
        to_include: list of algorithm names (abbreviations) which we wish to compare where
                    UPL   : approximation via uplinks
                    NRD   : approximation via naive rounding
                    IRD   : approximation via iterative rounding
                    MAT_B : approximation via matching heuristic, basic version: 
                            any max cardinality matching, random leaf-non-leaf selection
                    MAT_WG: approximation via matching heuristic with 
                            matching procedure max weight (W) and leaf-non-leaf selection via greedy algorithm (G)
                    MAT_WM: approximation via matching heuristic with
                            matching procedure maw weight (W) and leaf-non-leaf selection via leaf distance matching (M)
                    MAT_RG: approximation via matching heuristic with
                            matching procedure iterative removal (R) and leaf-non-leaf selection via greedy algorithm (G)
                    MAT_RM: approximation via matching heuristic with
                            matching procedure iterative removal (R) and leaf-non-leaf selection via leaf distance matching (M)
    Outputs:
        histogram of the approximation ratios of 500 randomly generated TAP instances with n nodes and factor*E links 
        for each of the specified algorithm.
    """
    axis.set_axisbelow(True)
    plt.sca(axis)
    plt.grid(visible = True, linewidth = 0.45, color = 'silver')
    

    samples = 500
    
    approx_ratios = []
    labels = []
    #for algorithm in to_include:
    if 'UPL' in to_include:
        ar_uplinks = upl.get_approximation_ratios_of_uplinks_approximation(samples, n, factor, OUTPUT_FOLDER)
        ar_uplinks = ar_uplinks[~np.isnan(ar_uplinks)]
        approx_ratios.append(ar_uplinks)
        labels.append('uplinks')
    if 'NRD' in to_include:
        ar_naiveRound = nrd.get_approximation_ratios_of_naiveRounding_approximation(samples, n, factor, OUTPUT_FOLDER)
        ar_naiveRound = ar_naiveRound[~np.isnan(ar_naiveRound)]
        approx_ratios.append(ar_naiveRound)
        labels.append('naive rounding')
    if 'IRD' in to_include:
        ar_iterativeRound = ird.get_approximation_ratios_of_iterativeRounding_approximation(samples, n, factor, OUTPUT_FOLDER)
        ar_iterativeRound = ar_iterativeRound[~np.isnan(ar_iterativeRound)]
        approx_ratios.append(ar_iterativeRound)
        labels.append('iterative rounding')
    if 'MAT_B' in to_include:
        ar_matching = mat_b.get_approximation_ratios_of_matchingHeuristic_approximation(samples, n, factor, OUTPUT_FOLDER)
        ar_matching = ar_matching[~np.isnan(ar_matching)]
        approx_ratios.append(ar_matching)
        labels.append('matching B')
    if 'MAT_WG' in to_include:
        ar_matching = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
            'max_weight', 'greedy', samples, n, factor, OUTPUT_FOLDER
        )
        ar_matching = ar_matching[~np.isnan(ar_matching)]
        approx_ratios.append(ar_matching)
        labels.append('matching WG')
    if 'MAT_WM' in to_include:
        ar_matching = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
            'max_weight', 'leaf_distance_matching', samples, n, factor, OUTPUT_FOLDER
        )
        ar_matching = ar_matching[~np.isnan(ar_matching)]
        approx_ratios.append(ar_matching)
        labels.append('matching WM')
    if 'MAT_RG' in to_include:
        ar_matching = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
            'iterative_removal', 'greedy', samples, n, factor, OUTPUT_FOLDER
        )
        ar_matching = ar_matching[~np.isnan(ar_matching)]
        approx_ratios.append(ar_matching)
        labels.append('matching RG')
    if 'MAT_RM' in to_include:
        ar_matching = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
            'iterative_removal', 'leaf_distance_matching', samples, n, factor, OUTPUT_FOLDER
        )
        ar_matching = ar_matching[~np.isnan(ar_matching)]
        approx_ratios.append(ar_matching)
        labels.append('matching RM')
        
    
    plt.hist(approx_ratios, label=labels, bins = 5)
    plt.legend(loc='upper right')
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    plt.xlabel('Approximation ratio')
    plt.ylabel('Number of occurrences')
    if factor == 'allLinks':
        plt.title(f'all links', loc = 'left')
    else:
        plt.title(f'$|L| = {factor} |E|$', loc = 'left')

        
        
        
        
        
        
def plot_average_approx_ratios(ns, factor, axis, to_include = ALL_ALGORITHMS[2:], linestyles = None):
    """
    Parameters:
        ns:         list of integers defining the number of nodes to consider
        factor:     defines how many links to generate. We generate factor * (# of edges of tree) links.
        axis:       axis on which to plot the historgram
        to_include: list of algorithm names (abbreviations) which we wish to compare where
                    UPL   : approximation via uplinks
                    NRD   : approximation via naive rounding
                    IRD   : approximation via iterative rounding
                    MAT_B : approximation via matching heuristic, basic version: 
                            any max cardinality matching, random leaf-non-leaf selection
                    MAT_WG: approximation via matching heuristic with 
                            matching procedure max weight (W) and leaf-non-leaf selection via greedy algorithm (G)
                    MAT_WM: approximation via matching heuristic with
                            matching procedure maw weight (W) and leaf-non-leaf selection via leaf distance matching (M)
                    MAT_RG: approximation via matching heuristic with
                            matching procedure iterative removal (R) and leaf-non-leaf selection via greedy algorithm (G)
                    MAT_RM: approximation via matching heuristic with
                            matching procedure iterative removal (R) and leaf-non-leaf selection via leaf distance matching (M)
    Outputs:
        histogram of the approximation ratios of 500 randomly generated TAP instances with n nodes and factor*E links 
        for each of the specified algorithm.
    """
    axis.set_axisbelow(True)
    plt.sca(axis)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    plt.grid(visible = True, linewidth = 0.45, color = 'silver')
    
    samples = 500
    
    if 'UPL' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = upl.get_approximation_ratios_of_uplinks_approximation(samples, n, factor, OUTPUT_FOLDER)
            avg_approx_ratios.append(np.nanmean(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('UPL')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'uplinks', linestyle = linestyle)
    if 'NRD' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = nrd.get_approximation_ratios_of_naiveRounding_approximation(samples, n, factor, OUTPUT_FOLDER)
            avg_approx_ratios.append(np.nanmean(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('NRD')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'naive rounding', linestyle = linestyle)
    if 'IRD' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = ird.get_approximation_ratios_of_iterativeRounding_approximation(samples, n, factor, OUTPUT_FOLDER)
            avg_approx_ratios.append(np.nanmean(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('IRD')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'iterative rounding', linestyle = linestyle)
    if 'MAT_B' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = mat_b.get_approximation_ratios_of_matchingHeuristic_approximation(samples, n, factor, OUTPUT_FOLDER)
            avg_approx_ratios.append(np.nanmean(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('MAT_B')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'matching B', linestyle = linestyle)
    if 'MAT_WG' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
                'max_weight', 'greedy', samples, n, factor, OUTPUT_FOLDER
            )
            avg_approx_ratios.append(np.nanmean(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('MAT_WG')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'matching WG', linestyle = linestyle)
    if 'MAT_WM' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
                'max_weight', 'leaf_distance_matching', samples, n, factor, OUTPUT_FOLDER
            )
            avg_approx_ratios.append(np.nanmean(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('MAT_WM')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'matching WM', linestyle = linestyle)
    if 'MAT_RG' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
                'iterative_removal', 'greedy', samples, n, factor, OUTPUT_FOLDER
            )
            avg_approx_ratios.append(np.nanmean(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('MAT_RG')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'matching RG', linestyle = linestyle)
    if 'MAT_RM' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
                'iterative_removal', 'leaf_distance_matching', samples, n, factor, OUTPUT_FOLDER
            )
            avg_approx_ratios.append(np.nanmean(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('MAT_RM')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'matching RM', linestyle = linestyle)
    
    
    plt.legend()
    plt.ylabel('Avg. approximation ratio')
    plt.xlabel('Tree size $n$')
    if factor == 'allLinks':
        plt.title(f'all links', loc = 'left')
    else:
        plt.title(f'|L| = ${factor}|E(T)|$', loc = 'left')
        
        
        
def plot_max_approx_ratios(ns, factor, axis, to_include = ALL_ALGORITHMS[2:], linestyles = None):
    """
    Parameters:
        ns:         list of integers defining the number of nodes to consider
        factor:     defines how many links to generate. We generate factor * (# of edges of tree) links.
        axis:       axis on which to plot the historgram
        to_include: list of algorithm names (abbreviations) which we wish to compare where
                    UPL   : approximation via uplinks
                    NRD   : approximation via naive rounding
                    IRD   : approximation via iterative rounding
                    MAT_B : approximation via matching heuristic, basic version: 
                            any max cardinality matching, random leaf-non-leaf selection
                    MAT_WG: approximation via matching heuristic with 
                            matching procedure max weight (W) and leaf-non-leaf selection via greedy algorithm (G)
                    MAT_WM: approximation via matching heuristic with
                            matching procedure maw weight (W) and leaf-non-leaf selection via leaf distance matching (M)
                    MAT_RG: approximation via matching heuristic with
                            matching procedure iterative removal (R) and leaf-non-leaf selection via greedy algorithm (G)
                    MAT_RM: approximation via matching heuristic with
                            matching procedure iterative removal (R) and leaf-non-leaf selection via leaf distance matching (M)
    Outputs:
        histogram of the approximation ratios of 500 randomly generated TAP instances with n nodes and factor*E links 
        for each of the specified algorithm.
    """
    axis.set_axisbelow(True)
    plt.sca(axis)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    plt.grid(visible = True, linewidth = 0.45, color = 'silver')
    
    samples = 500
    
    if 'UPL' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = upl.get_approximation_ratios_of_uplinks_approximation(samples, n, factor, OUTPUT_FOLDER)
            avg_approx_ratios.append(np.nanmax(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('UPL')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'uplinks', linestyle = linestyle)
    if 'NRD' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = nrd.get_approximation_ratios_of_naiveRounding_approximation(samples, n, factor, OUTPUT_FOLDER)
            avg_approx_ratios.append(np.nanmax(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('NRD')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'naive rounding', linestyle = linestyle)
    if 'IRD' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = ird.get_approximation_ratios_of_iterativeRounding_approximation(samples, n, factor, OUTPUT_FOLDER)
            avg_approx_ratios.append(np.nanmax(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('IRD')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'iterative rounding', linestyle = linestyle)
    if 'MAT_B' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = mat_b.get_approximation_ratios_of_matchingHeuristic_approximation(samples, n, factor, OUTPUT_FOLDER)
            avg_approx_ratios.append(np.nanmax(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('MAT_B')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'matching B', linestyle = linestyle)
    if 'MAT_WG' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
                'max_weight', 'greedy', samples, n, factor, OUTPUT_FOLDER
            )
            avg_approx_ratios.append(np.nanmax(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('MAT_WG')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'matching WG', linestyle = linestyle)
    if 'MAT_WM' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
                'max_weight', 'leaf_distance_matching', samples, n, factor, OUTPUT_FOLDER
            )
            avg_approx_ratios.append(np.nanmax(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('MAT_WM')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'matching WM', linestyle = linestyle)
    if 'MAT_RG' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
                'iterative_removal', 'greedy', samples, n, factor, OUTPUT_FOLDER
            )
            avg_approx_ratios.append(np.nanmax(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('MAT_RG')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'matching RG', linestyle = linestyle)
    if 'MAT_RM' in to_include:
        avg_approx_ratios = []
        for n in ns:
            approx_ratios = mat.get_approximation_ratios_of_matchingHeuristic_approximation(
                'iterative_removal', 'leaf_distance_matching', samples, n, factor, OUTPUT_FOLDER
            )
            avg_approx_ratios.append(np.nanmax(approx_ratios))
        if linestyles is not None:
            linestyle = linestyles[to_include.index('MAT_RM')]
        else:
            linestyle = '-'
        plt.plot(ns, avg_approx_ratios, label = 'matching RM', linestyle = linestyle)
    
    
    plt.legend()
    plt.ylabel('Max. approximation ratio')
    plt.xlabel('Tree size $n$')
    if factor == 'allLinks':
        plt.title(f'all links', loc = 'left')
    else:
        plt.title(f'|L| = ${factor}|E(T)|$', loc = 'left')