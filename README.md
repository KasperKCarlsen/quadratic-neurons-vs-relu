# Quadratic neurons vs. ReLU

This repository compares conventional ReLU networks with quadratic-neuron
networks, quadratic adapters, LoRA adapters, and related ViT experiments.

## Repository layout

- `py_files/` - the single canonical implementation of the CNN and ViT models.
- `HPO/` - Optuna training scripts, study inspection tools, and HPO job files.
- `Model_training/` - refitting/final-training code and CNN/adapter HPC jobs.
- `ViT/` - ViT training, refitting, pretrained-feature experiments, and jobs.
- `Visualizations/architecture/` - model architecture diagrams and their generator.
- `Results/repeated_seed_evaluation/Done_models/` - finalized model checkpoints,
  per-seed accuracy histories, and `seed_results.json` summaries.
- `Results/model_comparison/` - generated comparison tables from the finalized
  repeated-seed summaries.
- `Results/training_evaluation_visualizations/` - training/evaluation accuracy
  plots.

Generated checkpoints, Optuna databases, logs, caches, and local datasets are
ignored. The committed result artifacts are the finalized experiments intended
for analysis.

To regenerate the model comparison table:

```bash
python Results/model_comparison/summarize_results.py
```
