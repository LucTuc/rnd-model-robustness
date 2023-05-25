import numpy as np
import torch
import torch.nn as nn
import torchvision
import os
import itertools as it
from models import *
from attacks import *
from time import time
import argparse

available_experiments = ['non_adv_fc', 'adv_fc', 'non_adv_ensemble', 'adv_ensemble', 'non_adv_rse', 'adv_rse',
                         'non_adv_bnn', 'adv_bnn']

ens_mode = ['individual', 'collective']
eot_choices = [True, False]

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, choices=available_experiments,
                    help='Choose the experiment you wish to run')
parser.add_argument('--mode', required=False, type=str, choices=ens_mode, help='Choose a mode for ensemble training',
                    default=None)
parser.add_argument('--eot', action='store_true', help='Choose EOT attack or not')
args = parser.parse_args()

if torch.cuda.is_available():
    print('CUDA is available, training on GPU')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
train_loader_mnist = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./dataset/', train=True,
                                                                            transform=torchvision.transforms.ToTensor(),
                                                                            download=True),
                                                 batch_size=64, shuffle=True)
train_loader_fashion_mnist = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST('./dataset/', train=True,
                                                                                           transform=torchvision.transforms.ToTensor(),
                                                                                           download=True),
                                                         batch_size=64, shuffle=True)

# -------------------------- DEFINE THE EXPERIMENT SETUPS -------------------------- #

# GLOBAL SETUPS
eps_training = [0.075]  # [0.02, 0.05, 0.1]
pgd_steps_training = [10]  # [5, 10, 20]
training_epochs = [1, 5, 10, 20, 50]
data_set_names = ['mnist', 'fashion_mnist']
data_set_train_loaders = {'mnist': train_loader_mnist,
                          'fashion_mnist': train_loader_fashion_mnist
                          }

# SETUPS FOR ALL SINGLE BASE ARCHITECTURES
architecture_layouts = {1: [100, 10],
                        2: [200, 200, 10],
                        3: [300, 300, 300, 10],
                        4: [400, 400, 400, 400, 10]
                        }
architectures = [1, 2, 3, 4]

# INDIVIDUAL SETUPS
# Simple fully connected --> no additional setups
# RSE and BNN
# eot = ['True', 'False']
eot_steps = [10]  # [10, 20, 50]
# Ensemble
# mode = ['individual', 'collective']
collective_architectures = [[[100, 10], [200, 10], [50, 100, 10]],
                            [[300, 200, 10], [500, 200, 10], [500, 500, 200, 10]],
                            [[100, 10], [200, 10], [200, 100, 10], [300, 200, 100, 10], [500, 200, 10]],
                            [[500, 300, 200, 10], [500, 500, 200, 10], [500, 400, 300, 200, 100, 10],
                             [512, 1024, 512, 256, 10], [512, 512, 256, 256, 10]]]

# -------------------------- RUN THE EXPERIMENTS -------------------------- #

# SIMPLE FULLY CONNECTED
# No adversarial training
if args.experiment == 'non_adv_fc':
    iter_prod_fc_not_adversarial = it.product(architectures, data_set_names)
    comb_len = len(architectures) * len(data_set_names)
    print('Fully Connected NN')
    print('Traditional Training')
    start = time()
    for i, arg_tup in enumerate(iter_prod_fc_not_adversarial):
        print(f'Progress: {np.around(i / comb_len * 100, 2)}%    Time elapsed: {np.around(time() - start, 1)}s',
              end='\r')
        folder_prompt = f'fc_nonadv_{arg_tup[1]}_arch{arg_tup[0]}'
        os.system('mkdir -p pre_trained_models/fc_nets/' + folder_prompt)
        if arg_tup[1] == 'mnist':
            fc_net = FullyConnected(fc_layers=architecture_layouts[arg_tup[0]], input_size=28, device=DEVICE,
                                    data_mean=0.1307, data_std=0.3081).to(DEVICE)
        elif arg_tup[1] == 'fashion_mnist':
            fc_net = FullyConnected(fc_layers=architecture_layouts[arg_tup[0]], input_size=28, device=DEVICE,
                                    data_mean=0.2860, data_std=0.3530).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(fc_net.parameters())
        fc_net_trainer = FCNetTrainer(train_loader=data_set_train_loaders[arg_tup[1]], n_epochs=50, criterion=criterion,
                                      optimizer=optimizer, device=DEVICE, verbose=False)
        if os.path.isfile(f'pre_trained_models/fc_nets/{folder_prompt}/{folder_prompt}_epoch{training_epochs[-1]}.pt'):
            continue
        else:
            fc_net_trainer.train(fc_net, checkpoints=training_epochs,
                                 save_path_base=f'pre_trained_models/fc_nets/{folder_prompt}/{folder_prompt}')

# adversarial training
elif args.experiment == 'adv_fc':
    iter_prod_fc_adversarial = it.product(architectures, data_set_names, eps_training, pgd_steps_training)
    comb_len = len(architectures) * len(data_set_names) * len(eps_training) * len(pgd_steps_training)
    print('Fully Connected NN')
    print('Adversarial Training')
    start = time()
    for i, arg_tup in enumerate(iter_prod_fc_adversarial):
        print(f'Progress: {np.around(i / comb_len * 100, 2)}%    Time elapsed: {np.around(time() - start, 1)}s',
              end='\r')
        folder_prompt = f'fc_adv_{arg_tup[1]}_arch{arg_tup[0]}_eps{arg_tup[2]}_steps{arg_tup[3]}'
        os.system('mkdir -p pre_trained_models/fc_nets/' + folder_prompt)
        if arg_tup[1] == 'mnist':
            fc_net = FullyConnected(fc_layers=architecture_layouts[arg_tup[0]], input_size=28, device=DEVICE,
                                    data_mean=0.1307, data_std=0.3081).to(DEVICE)
        elif arg_tup[1] == 'fashion_mnist':
            fc_net = FullyConnected(fc_layers=architecture_layouts[arg_tup[0]], input_size=28, device=DEVICE,
                                    data_mean=0.2860, data_std=0.3530).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(fc_net.parameters())
        fc_net_trainer = FCNetTrainer(train_loader=data_set_train_loaders[arg_tup[1]], n_epochs=50, criterion=criterion,
                                      optimizer=optimizer, device=DEVICE, verbose=False)
        attack = PGD(epsilon=arg_tup[2], n_iterations=arg_tup[3], step_size=0.05)
        if os.path.isfile(f'pre_trained_models/fc_nets/{folder_prompt}/{folder_prompt}_epoch{training_epochs[-1]}.pt'):
            continue
        else:
            fc_net_trainer.adversarial_train(fc_net, attack=attack, checkpoints=training_epochs,
                                             save_path_base=f'pre_trained_models/fc_nets/{folder_prompt}/{folder_prompt}')

# ENSEMBLE OF FULLY CONNECTED
# no adversarial training
elif args.experiment == 'non_adv_ensemble':
    iter_prod_ensemble_non_adversarial = it.product(architectures, data_set_names)
    comb_len = len(architectures) * len(data_set_names)
    print('Ensemble of Fully Connected')
    print('Traditional Training')
    start = time()
    for i, arg_tup in enumerate(iter_prod_ensemble_non_adversarial):
        print(f'Progress: {np.around(i / comb_len * 100, 2)}%    Time elapsed: {np.around(time() - start, 1)}s',
              end='\r')
        folder_prompt = f'ensemble_nonadv_{arg_tup[1]}_archcol{arg_tup[0]}'
        os.system('mkdir -p pre_trained_models/fc_ensemble_nets/' + folder_prompt)
        if arg_tup[1] == 'mnist':
            ensemble_net = EnsembleOfFullyConnected(device=DEVICE, input_size=28,
                                                    member_layouts=collective_architectures[arg_tup[0]-1],
                                                    data_mean=0.1307, data_std=0.3081).to(DEVICE)
        elif arg_tup[1] == 'fashion_mnist':
            ensemble_net = EnsembleOfFullyConnected(device=DEVICE, input_size=28,
                                                    member_layouts=collective_architectures[arg_tup[0]-1],
                                                    data_mean=0.2860, data_std=0.3530).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizers = [torch.optim.Adam(member.parameters()) for member in ensemble_net.members]
        ensemble_net_trainer = EnsembleOfFCTrainer(train_loader=data_set_train_loaders[arg_tup[1]], criterion=criterion,
                                                   optimizers=optimizers, n_epochs=50, device=DEVICE, verbose=False)
        if os.path.isfile(f'pre_trained_models/fc_ensemble_nets/{folder_prompt}/{folder_prompt}_epoch{training_epochs[-1]}.pt'):
            continue
        else:
            ensemble_net_trainer.train(ensemble_net, checkpoints=training_epochs,
                                       save_path_base=f'pre_trained_models/fc_ensemble_nets/{folder_prompt}/{folder_prompt}')

# with adversarial training
elif args.experiment == 'adv_ensemble':
    iter_prod_ensemble_adversarial = it.product(architectures, data_set_names, eps_training, pgd_steps_training)
    comb_len = len(architectures) * len(data_set_names) * len(eps_training) * len(pgd_steps_training)
    print('Ensemble of Fully Connected')
    print('Adversarial Training')
    start = time()
    for i, arg_tup in enumerate(iter_prod_ensemble_adversarial):
        print(f'Progress: {np.around(i / comb_len * 100, 2)}%    Time elapsed: {np.around(time() - start, 1)}s',
              end='\r')
        folder_prompt = f'ensemble_adv_{arg_tup[1]}_archcol{arg_tup[0]}_eps{arg_tup[2]}_steps{arg_tup[3]}_mode{args.mode}'
        os.system('mkdir -p pre_trained_models/fc_ensemble_nets/' + folder_prompt)
        if arg_tup[1] == 'mnist':
            ensemble_net = EnsembleOfFullyConnected(device=DEVICE, input_size=28,
                                                    member_layouts=collective_architectures[arg_tup[0]-1],
                                                    data_mean=0.1307, data_std=0.3081).to(DEVICE)
        elif arg_tup[1] == 'fashion_mnist':
            ensemble_net = EnsembleOfFullyConnected(device=DEVICE, input_size=28,
                                                    member_layouts=collective_architectures[arg_tup[0]-1],
                                                    data_mean=0.2860, data_std=0.3530).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizers = [torch.optim.Adam(member.parameters()) for member in ensemble_net.members]
        ensemble_net_trainer = EnsembleOfFCTrainer(train_loader=data_set_train_loaders[arg_tup[1]], criterion=criterion,
                                                   optimizers=optimizers, n_epochs=50, device=DEVICE, verbose=False)
        attack = PGD(epsilon=arg_tup[2], n_iterations=arg_tup[3], step_size=0.05)
        if os.path.isfile(f'pre_trained_models/fc_ensemble_nets/{folder_prompt}/{folder_prompt}_epoch{training_epochs[-1]}.pt'):
            continue
        else:
            ensemble_net_trainer.adversarial_train(ensemble_net, attack=attack, mode=args.mode, checkpoints=training_epochs,
                                                   save_path_base=f'pre_trained_models/fc_ensemble_nets/{folder_prompt}/{folder_prompt}')

# RANDOM SELF ENSEMBLES
# no adversarial training
elif args.experiment == 'non_adv_rse':
    iter_prod_rse_not_adversarial = it.product(architectures, data_set_names)
    comb_len = len(architectures) * len(data_set_names)
    print('Random Self Ensemble')
    print('Traditional Training')
    start = time()
    for i, arg_tup in enumerate(iter_prod_rse_not_adversarial):
        print(f'Progress: {np.around(i / comb_len * 100, 2)}%    Time elapsed: {np.around(time() - start, 1)}s',
              end='\r')
        folder_prompt = f'rse_nonadv_{arg_tup[1]}_arch{arg_tup[0]}'
        os.system('mkdir -p pre_trained_models/rse_nets/' + folder_prompt)
        if arg_tup[1] == 'mnist':
            rse_net = FullyConnectedRandomSelfEnsemble(fc_layers=architecture_layouts[arg_tup[0]], input_size=28,
                                                       device=DEVICE, std_init=0.01, std_inner=10, data_mean=0.1307,
                                                       data_std=0.3081).to(DEVICE)
        elif arg_tup[1] == 'fashion_mnist':
            rse_net = FullyConnectedRandomSelfEnsemble(fc_layers=architecture_layouts[arg_tup[0]], input_size=28,
                                                       device=DEVICE, std_init=0.01, std_inner=10, data_mean=0.2860,
                                                       data_std=0.3530).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(rse_net.parameters())
        rse_net_trainer = FullyConnectedRSETrainer(train_loader=data_set_train_loaders[arg_tup[1]], n_epochs=50,
                                                   criterion=criterion,
                                                   optimizer=optimizer, device=DEVICE, verbose=False)
        if os.path.isfile(f'pre_trained_models/rse_nets/{folder_prompt}/{folder_prompt}_epoch{training_epochs[-1]}.pt'):
            continue
        else:
            rse_net_trainer.train(rse_net, checkpoints=training_epochs,
                                  save_path_base=f'pre_trained_models/rse_nets/{folder_prompt}/{folder_prompt}')

# with adversarial training
elif args.experiment == 'adv_rse':
    iter_prod_rse_adversarial = it.product(architectures, data_set_names, eps_training, pgd_steps_training)
    comb_len = len(architectures) * len(data_set_names) * len(eps_training) * len(pgd_steps_training)
    print('Random Self Ensemble')
    print('Adversarial Training')
    start = time()
    for i, arg_tup in enumerate(iter_prod_rse_adversarial):
        print(f'Progress: {np.around(i / comb_len * 100, 2)}%    Time elapsed: {np.around(time() - start, 1)}s',
              end='\r')
        folder_prompt = f'rse_adv_{arg_tup[1]}_arch{arg_tup[0]}_eps{arg_tup[2]}_steps{arg_tup[3]}_eot{str(args.eot)}'
        os.system('mkdir -p pre_trained_models/rse_nets/' + folder_prompt)
        if arg_tup[1] == 'mnist':
            rse_net = FullyConnectedRandomSelfEnsemble(fc_layers=architecture_layouts[arg_tup[0]], input_size=28,
                                                       device=DEVICE, std_init=0.01, std_inner=10, data_mean=0.1307,
                                                       data_std=0.3081).to(DEVICE)
        elif arg_tup[1] == 'fashion_mnist':
            rse_net = FullyConnectedRandomSelfEnsemble(fc_layers=architecture_layouts[arg_tup[0]], input_size=28,
                                                       device=DEVICE, std_init=0.01, std_inner=10, data_mean=0.2860,
                                                       data_std=0.3530).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(rse_net.parameters())
        rse_net_trainer = FullyConnectedRSETrainer(train_loader=data_set_train_loaders[arg_tup[1]], n_epochs=50,
                                                   criterion=criterion,
                                                   optimizer=optimizer, device=DEVICE, verbose=False)
        attack = PGD(epsilon=arg_tup[2], n_iterations=arg_tup[3], step_size=0.05, EOT=args.eot)
        if os.path.isfile(f'pre_trained_models/rse_nets/{folder_prompt}/{folder_prompt}_epoch{training_epochs[-1]}.pt'):
            continue
        else:
            rse_net_trainer.adversarial_train(rse_net, attack=attack, checkpoints=training_epochs,
                                              save_path_base=f'pre_trained_models/rse_nets/{folder_prompt}/{folder_prompt}')

# FULLY CONNECTED BAYESIAN NNs
# no adversarial training
elif args.experiment == 'non_adv_bnn':
    iter_prod_bnn_not_adversarial = it.product(architectures, data_set_names)
    comb_len = len(architectures) * len(data_set_names)
    print('Bayesian Neural Network')
    print('Traditional Training')
    start = time()
    for i, arg_tup in enumerate(iter_prod_bnn_not_adversarial):
        print(f'Progress: {np.around(i / comb_len * 100, 2)}%    Time elapsed: {np.around(time() - start, 1)}s',
              end='\r')
        folder_prompt = f'bnn_nonadv_{arg_tup[1]}_arch{arg_tup[0]}'
        os.system('mkdir -p pre_trained_models/bnn_nets/' + folder_prompt)
        if arg_tup[1] == 'mnist':
            bnn_net = BayesianFullyConnected(fc_layers=architecture_layouts[arg_tup[0]], input_size=28, device=DEVICE,
                                             data_mean=0.1307, data_std=0.3081).to(DEVICE)
        elif arg_tup[1] == 'fashion_mnist':
            bnn_net = BayesianFullyConnected(fc_layers=architecture_layouts[arg_tup[0]], input_size=28, device=DEVICE,
                                             data_mean=0.2860, data_std=0.3530).to(DEVICE)
        optimizer = torch.optim.Adam(bnn_net.parameters())
        bnn_net_trainer = FullyConnectedBNNTrainer(train_loader=data_set_train_loaders[arg_tup[1]], n_epochs=50,
                                                   optimizer=optimizer, device=DEVICE, verbose=False)
        if os.path.isfile(f'pre_trained_models/bnn_nets/{folder_prompt}/{folder_prompt}_epoch{training_epochs[-1]}.pt'):
            continue
        else:
            bnn_net_trainer.train(bnn_net, checkpoints=training_epochs,
                                  save_path_base=f'pre_trained_models/bnn_nets/{folder_prompt}/{folder_prompt}')

# with adversarial training
elif args.experiment == 'adv_bnn':
    iter_prod_bnn_adversarial = it.product(architectures, data_set_names, eps_training, pgd_steps_training)
    comb_len = len(architectures) * len(data_set_names) * len(eps_training) * len(pgd_steps_training)
    print('Bayesian Neural Network')
    print('Adversarial Training')
    start = time()
    for i, arg_tup in enumerate(iter_prod_bnn_adversarial):
        print(f'Progress: {np.around(i / comb_len * 100, 2)}%    Time elapsed: {np.around(time() - start, 1)}s',
              end='\r')
        folder_prompt = f'bnn_adv_{arg_tup[1]}_arch{arg_tup[0]}_eps{arg_tup[2]}_steps{arg_tup[3]}_eot{str(args.eot)}'
        os.system('mkdir -p pre_trained_models/bnn_nets/' + folder_prompt)
        if arg_tup[1] == 'mnist':
            bnn_net = BayesianFullyConnected(fc_layers=architecture_layouts[arg_tup[0]], input_size=28, device=DEVICE,
                                             data_mean=0.1307, data_std=0.3081).to(DEVICE)
        elif arg_tup[1] == 'fashion_mnist':
            bnn_net = BayesianFullyConnected(fc_layers=architecture_layouts[arg_tup[0]], input_size=28, device=DEVICE,
                                             data_mean=0.2860, data_std=0.3530).to(DEVICE)
        optimizer = torch.optim.Adam(bnn_net.parameters())
        bnn_net_trainer = FullyConnectedBNNTrainer(train_loader=data_set_train_loaders[arg_tup[1]], n_epochs=50,
                                                   optimizer=optimizer, device=DEVICE, verbose=False)
        attack = PGD(epsilon=arg_tup[2], n_iterations=arg_tup[3], step_size=0.05, EOT=args.eot)
        if os.path.isfile(f'pre_trained_models/bnn_nets/{folder_prompt}/{folder_prompt}_epoch{training_epochs[-1]}.pt'):
            continue
        else:
            bnn_net_trainer.adversarial_train(bnn_net, attack=attack, checkpoints=training_epochs,
                                              save_path_base=f'pre_trained_models/bnn_nets/{folder_prompt}/{folder_prompt}')

else:
    raise RuntimeError('If you read this message then something must have gone horribly wrong')
