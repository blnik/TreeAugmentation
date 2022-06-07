# TreeAugmentation
This repository contains all code used throughout the semester project on the Tree Augmentation Problem. It contains function definitions, numerical simulations, and images produced for the report.

## Necessary Modules
The following modules are necessary to run the code (the versions specified are the versions used by the author).
- gurobipy 9.5.1 (for guidance on installation and licensing, I recommend following [this](https://www.youtube.com/watch?v=fRKhao2bzsY%5D) video.)
- matplotlib 3.4.2
- networkx 2.6.2
- numpy 1.21.0
- pandas 1.3.0
- pydot 1.4.2
- tqdm 4.63.0

## Project Description
### The `tree_augmentation` module
The `tree_augmentation` module contains a set of sub-modules which store functions connected to the Tree Augmentation Problem.
- In `general.py`, we define a number of functions that can be used to define and draw TAP instances as well as do basic operations on TAP instances.
- In `natural_lp.py`, we define functions that are related to the LP-formulation of the TAP.
- In `approximation_naiveRounding.py`, we define functions that are related to the approximation algorithm in which we round up every non-zero element as outputted by the natural LP relaxation.
- In `approximation_iterativeRounding.py`, we define functions that are related to the approximation algorithm that iteratively rounds the solution of the natural LP relaxation.
- In `approximation_uplinks.py`, we define functions that are related to the approximation algorithm which replaces every link that is not an uplink by two uplinks and computes the natural LP relaxation on the newly obtained TAP instance (which is integral).
- In `approximation_matching_heuristic_basic.py`, we define functions that are related to the matching heuristic in its simplest form.
- In `approximation_matching_heuristic_basic.py`, we define functions that are related to the matching heuristic after adding a couple of improvements.
- In `comparisons.py`, we define functions that are related to comparing the different algorithms with each other.

### The `numerical_simulations.ipynb` notebook
Refer to this notebook for a gentle introduction to the functions defined in the `tree_augmentation` module. Here, example uses of the most important functions are presented.

### The `numerical_simulations.ipynb` notebook
Here we simulate a lot of instances and let the different approximation algorithms try to solve the simulated instances. The results are shown in neat plots.

### The rest of the files
- Whenever we simulate a lot of instances, we save their results as NumPy arrays in the `saved_outputs` folder.
- All images that are produced to be included in the report are stored in the `report_images` folder.
- The `draw_example_graphs` folder contains an array of notebooks that draw TAP instances which are used in the report
