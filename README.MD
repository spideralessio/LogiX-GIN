# On Logic-based Self-Explainable Graph Neural Networks - LogiX-GIN
Official repository of [LogiX-GIN](https://neurips.cc/virtual/2025/poster/118261), accepted at NeurIPS 2025.

To cite our work, please use:
```
@inproceedings{ragno2025,
  title={On Logic-based Self-Explainable Graph Neural Networks},
  author={Ragno, Alessio and Plantevit, Marc and Robardet, Celine},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

The environment details are available in the `environment.yml` file.

## Training Models:
It is possible to train the models using the following commands
- Black-box models
`python train_baseline.py` and `python train_baseline_node.py` for a single run or `python optimize_baseline.py` and `python optimize_baseline_node.py` to do a grid search
- Logic models
`python train_logic.py` and `python train_logic_node.py` for a single run or `python optimize_logic.py` and `python optimize_logic_node.py` to do a grid search

## Analyzing results:
- `nbs/Results.ipynb` allows visualizing classification results
- `nbs/InstanceLevelExpl.ipynb` allows obtaining node level explanation and calculating fidelity
- `nbs/PatternAnalysis.ipynb` allows studying the effects of pruning
- `nbs/LayerWiseRules.ipynb` allows obtaining layerwise rules of the models

## Example:
The repository contains a simple example of a model trained on MUTAG with 2 layers using seed 0. This example was obtained by running the following commands: 
- `python train_baseline.py --dataset MUTAG --seed 0 --epochs 3000 --batch_size 128 --hidden_dim 32 --num_layers 2 --dropout 0.5 --l2 0.0001 --nogumbel --lr 0.001`
- `python train_baseline.py --dataset MUTAG --seed 0 --epochs 3000 --batch_size 32 --hidden_dim 32 --num_layers 2 --l2 0.0 --lr 0.01`
- `python train_logic.py --dataset MUTAG --seed 0 --epochs 5000 --batch_size 32 --l2 0.0 --lr 0.01 --fc_reg 0.01 --conv_reg 0.001`



