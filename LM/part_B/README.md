# NLU Course Project - Part 1: LM
This folder contains the code necessary for running the part 1-B of the NLU course project at the University of Trento, focused on next-word prediction.

More details about the project, implementation and results, are provided in the report: [LM Report](../LM_report.pdf)

## Setting up Experiments
This project uses [**Hydra**](https://hydra.cc/) to manage hyperparameters configurations to test and keep track of the results of each configuration.

To modify the search space for a specific part, it is necessary to modify the corresponding config file.

```text
├── configs/
│   ├── config.yaml         # Global config parameters, results folder paths and logging formats
│   └── part/
│       ├── 1b0.yaml        # Config file for Baseline LSTM
│       ├── 1b1.yaml        # Config file for Weight Tying
│       ├── 1b2.yaml        # Config file for Variational Dropout
│       └── 1b3.yaml        # Config file for Non-monotonically Triggered AvSGD
```

Inside each part's config file there are two sets of hyperparameters, `parameters` and `best_parameters`. The first one defines the search space when testing new hyperparameters, and the second holds the hyperparameters of the current best-performing model. This is used to keep track of the optimal configuration and to create the dummy model in which to load the saved weights during evaluation.

Hydra saves the result in the following format
```text
├── results/
│   ├── partXXX/
│   │   ├── best_model/                           # Final evaluated best model 
│   │   │   ├── loss_plot.png                     # Loss plot for best model
│   │   │   ├── losses.json                       # Losses and hyperparams for best model
│   │   │   └── model.pt                          # Best model saved, used when testing is false
│   │   └── sweep_YYYY-MM-DD_HH-MM-SS/            # Sweep folder with date
│   │       ├── best_model/                       # Best model found during this specific sweep
│   │       ├── hid_size=XXX_emb_size=XXX_lr=XXX/ # Results for this particular trial/configuration
│   │       ├── main.log                          # Code output logs
│   │       └── sweep_summary.json                # Summary of all the runs
│   └── partXXX/
```

## Code Usage
After configuring your parameter space or selecting which model to evaluate, the code is ready to run.

To execute the pipeline, you must pass a `part` value to specify the model architecture:

- 1b0 for Baseline LSTM
- 1b1 for Weight Tying
- 1b2 for Variational Dropout
- 1b3 for Non-monotonically Triggered AvSGD

You must also pass a `testing` boolean to set the execution mode:

- `true` to run an [**Optuna**](https://optuna.org/) hyperparameter sweep
- `false` to evaluate the best saved model. (The script will automatically load the weights from the corresponding results/partXXX/best_model/model.pt file shown in the directory tree above).

**Example command:**
```bash
uv run python main.py part=1b1 testing=false
```