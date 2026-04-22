# Training Workspace

This directory contains training and model lifecycle assets.

- data/raw: Raw collected or downloaded clips
- data/processed: Normalized frame/sequence datasets
- models/checkpoints: Intermediate training checkpoints
- models/exported: Versioned deployment-ready model binaries
- configs: Experiment and hyperparameter configuration
- scripts: Data prep, training, and evaluation scripts

Keep training code isolated from runtime services for clean deployment boundaries.