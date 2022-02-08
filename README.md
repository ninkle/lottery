# lottery

This is a basic repository for running experiments related to the Lottery Ticket Hypothesis. More recently, this repository has been used for demonstrating a set of basic code patterns for machine learning experiments. 

The main components include:

1. A `model.py` file, which includes all `Model` objects.
2. A `trainer.py` file, which includes all `Trainer` objects - a `Trainer` typically implements a Pytorch Dataloader (if using a native dataset) and the training, test, and validation loops.
3. A `train.py` file, which acts as the run script and manages most loading and saving actions an `hps.yaml` file, through which experiment hyperparameters are specified.

Users will want to install the PyYAML (`pip install PyYAML`) and tqdm (`pip install tqdm`) packages.
