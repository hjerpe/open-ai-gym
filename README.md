# Experiment with RL using OpenAI's gym library

## Setting up environment and installing packages

### Create conda environment
```bash
conda create -n lunar-lander python=3.10
```

### Install packages
Set up an environment using either conda or pyenv and run
```bash
conda activate lunar-lander
pip install -r requirements.txt
```

**Note** `swig` is required for `gym[box2d]` and is not always installed by default. There are a number of ways to install it, google is your friend.

## Logging with MLflow
To log runs with MLflow use the `--mlflow-logging`-flag and make sure the following environment variables set:

* `MLFLOW_TRACKING_URI`
* `MLFLOW_TRACKING_USERNAME`
* `MLFLOW_TRACKING_PASSWORD`

Use `--mlflow-experiment-name` and `--mlflow-run-name` to set experiment and run name. 

Learn more, https://www.mlflow.org/docs/latest/tracking.html
