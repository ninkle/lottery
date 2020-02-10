import torch
import torch.nn as nn
import click
import os

from model import Net
from trainer import Trainer 

@click.group()
def cli():
    pass

@cli.command()
@click.option("--epochs", default=10)
@click.option("--lr", default=0.001)
@click.option("--gamma", default=0.7)
@click.option("--batch-size", default=32)
@click.option("--experiment-name", type=str, help="give your experiment a unique name")
def train_dense_model(experiment_name, epochs, lr, gamma, batch_size):
    
    save_dir = os.path.join("experiments", experiment_name)

    model = Net()
    trainer = Trainer(model, batch_size, epochs, lr, gamma, save_dir)

    # save initialization scheme
    init_path = os.path.join(save_dir, "init.pt")
    model.save(init_path)

    # train model - the model will save in experiments/$EXPERIMENT_DIR/trained.pt 
    trainer.train()

@cli.command()
@click.option("--pruning-percentage", default=0.7)
@click.option("--pruning-method", default="layerwise", help="options: `layerwise` or `global`")
@click.option("--experiment-name", type=str, help="parent directory where a trained model is stored")
@click.option("--save_file", default="pruned.pt")
@click.option("--epochs", default=10)
@click.option("--lr", default=0.001)
@click.option("--gamma", default=0.7)
@click.option("--batch_size", default=32)
def train_pruned_model(pruning_percentage, pruning_method, experiment_name, save_file, epochs, lr, gamma, batch_size):
    
    save_dir = os.path.join("experiments", experiment_name)

    model_train_path = os.path.join(save_dir, "trained.pt")
    model_init_path = os.path.join(save_dir, "init.pt")

    trained_model = Net()
    init_model = Net()

    # load trained model
    trained_state_dict = torch.load(model_train_path)
    trained_model.load_state_dict(trained_state_dict)

    # load model's init state
    init_state_dict = torch.load(model_init_path)
    init_model.load_state_dict(init_state_dict)

    # prune trained model and generate weight mask
    mask = trained_model.prune(pruning_method, pruning_percentage)

    # mask init model 
    init_model.mask(mask)

    # train pruned model - the model will save in $EXPERIMENT_DIR/pruned.pt
    trainer = Trainer(init_model, batch_size, epochs, lr, gamma, save_dir, pruned=True)
    trainer.train()

if __name__ == "__main__":
    cli()

    
    






