# Official Repository of LogiX-GIN
The environment details are available in the `environment.yml` file.

## Training Models:
We provide best weights in folders results and results_logic, however, it is possible to train the models using the following commands
- Black-box models
`python train_baseline.py` and `python train_baseline_node.py` for a single run or `python optimize_baseline.py` and `python optimize_baseline_node.py` to do a grid search
- Logic models
`python train_logic.py` and `python train_logic_node.py` for a single run or `python optimize_logic.py` and `python optimize_logic_node.py` to do a grid search

## Analyzing results:
Files `Results.ipynb` and `ResultsExpl.ipynb` can be used to generate plots and tables from the paper

## Node Attributions:
Node attributions can be done with the file `InstanceLevelExpl.ipynb`

## Global Explanation and Layer-wise Rules:
Global Explanation and Layer-wise Rules are obtainable on the `GlobalExplanationsGraph` and `GlobalExplanationsNode` files

