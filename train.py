import torch
import torch.nn as nn

import yaml
import os

from model import Net
from trainer import Trainer 


def train_dense_model(config):
    ############# unpack config ######################
    directory_name = config["save_directory"]
    experiment_name = config["save_experiment"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    gamma = config["gamma"]
    device = config["device"]
    ###################################################

    # init model, trainer
    save_dir = os.path.join("dense_experiments", directory_name, experiment_name)
    model = Net()
    trainer = Trainer(model, batch_size, epochs, lr, gamma, save_dir, device=device)

    # save initialization scheme
    init_path = os.path.join(save_dir, "init.pt")
    model.save(init_path)

    # train model
    trainer.train()

    # save hps and metadata
    with open(os.path.join(save_dir, f"{experiment_name}_hps.yaml"), "w") as f:
        yaml.dump(config, f)

    ####################################################
    # model saves at $/Research/Lottery/dense_experiments/$DIR_NAME/$EXP_NAME/trained.pt
    # init saves at $/Research/Lottery/dense_experiments/$DIR_NAME/$EXP_NAME/init.pt
    # hps saves at $/Research/Lottery/dense_experiments/$DIR_NAME/$EXP_NAME/$EXP_NAME_hps.yaml
    ####################################################


def train_pruned_model(config):
    ############# unpack config ######################
    load_directory = config["load_directory"]
    load_experiment = config["load_experiment"]
    save_directory = config["save_directory"]
    save_experiment = config["save_experiment"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    gamma = config["gamma"]
    pruning_method = config["pruning_method"]
    pruning_percentage = config["pruning_percentage"]
    device = config["device"]
    ###################################################
    
    load_dir = os.path.join("dense_experiments", load_directory, load_experiment)
    save_dir = os.path.join("pruned_experiments", save_directory, save_experiment)
    model_train_path = os.path.join(load_dir, "trained.pt") 
    model_init_path = os.path.join(load_dir, "init.pt") 

    # init models
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
    trainer = Trainer(init_model, batch_size, epochs, lr, gamma, save_dir, device=device, pruned=True)
    trainer.train()

    # save hps and metadata
    with open(os.path.join(save_dir, f"{save_experiment}_hps.yaml"), "w") as f:
        yaml.dump(config, f)

    ####################################################
    # model saves at $/Research/Lottery/pruned_experiments/$DIR_NAME/$EXP_NAME/trained.pt
    # init saves at $/Research/Lottery/pruned_experiments/$DIR_NAME/$EXP_NAME/init.pt
    # hps saves at $/Research/Lottery/pruned_experiments/$DIR_NAME/$EXP_NAME/$EXP_NAME_hps.yaml
    ####################################################


if __name__ == "__main__":
    with open("hps.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # choose train script
    train_pruned_model(config["prune"])
    #train_dense_model(config["dense"])
    

    

    
    






