# Quadratic neurons vs. ReLU

This repository stems from a project work course on the Technical University of Denmark (DTU) and compares conventional ReLU networks with quadratic-neuron networks, quadratic adapters with LoRA adapters, and related ViT experiments.

This was used to test whether non-linear activation functions can be omitted from quadratic networks, in particular quadratic convolutional neural networks (QCNN), without hurting the performance of the models. 

## Repository layout

- `py_files/` - the implementation of the CNN and ViT models.
- `HPO/` - Optuna training scripts, study inspection tools, and HPO job files.
- `Model_training/` - refitting/final-training code and CNN/adapter HPC jobs.
- `ViT/` - ViT training, refitting, pretrained-feature experiments, and jobs.
- `Visualizations/` - visualizations of the architectures used.
- `Results/repeated_seed_evaluation/Done_models/` - finalized model checkpoints,
  per-seed accuracy histories, and `seed_results.json` summaries.
- `Results/model_comparison/` - generated comparison tables from the finalized
  repeated-seed summaries.
- `Results/training_evaluation_visualizations/` - training/evaluation accuracy
  plots.

Intermediate results such as checkpoints, logs, caches, e.g. are left out to preserve readability in the repo. \
Final model accuracies, comparisons and training visualizations can be found in the Results directory.


To update model_comparison.csv and model_comparison.json given new results:

```bash
python Results/model_comparison/summarize_results.py
```
