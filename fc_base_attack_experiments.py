import numpy as np
import torch
import os
import torchvision
import pandas as pd
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
parser.add_argument('--no_warm_start', action='store_false', help='Continue evaluation from previous results')
args = parser.parse_args()

if torch.cuda.is_available():
    print('CUDA is available, training on GPU')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
test_loader_mnist = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./dataset/', train=False,
                                                                           transform=torchvision.transforms.ToTensor(),
                                                                           download=True),
                                                batch_size=1000, shuffle=True)

test_loader_fashion_mnist = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST('./dataset/', train=False,
                                                                                   transform=torchvision.transforms.ToTensor(),
                                                                                   download=True),
                                                        batch_size=1000, shuffle=True)

header = ['net_type', 'net_num', 'data_set', 'n_epochs', 'training', 'eps_train', 'pgd_steps_train', 'eot_train',
          'eot_steps_train', 'eps_test', 'pgd_steps_test', 'eot_test', 'eot_steps_test', 'test_acc',
          'adv_acc']

# -------------------------- RECALL THE PRETRAIN SETUPS -------------------------- #

# GLOBAL SETUPS
eps_training = [0.075]  # [0.02, 0.05, 0.1]
pgd_steps_training = [10]  # [5, 10, 20]
training_epochs = [1, 5, 10, 20, 50]
data_set_names = ['mnist', 'fashion_mnist']
data_set_test_loaders = {'mnist': test_loader_mnist,
                         'fashion_mnist': test_loader_fashion_mnist
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
eot_steps_training = [10]  # [10, 20, 50]
# Ensemble
# mode = ['individual', 'collective']
collective_architectures = [[[100, 10], [200, 10], [50, 100, 10]],
                            [[300, 200, 10], [500, 200, 10], [500, 500, 200, 10]],
                            [[100, 10], [200, 10], [200, 100, 10], [300, 200, 100, 10], [500, 200, 10]],
                            [[500, 300, 200, 10], [500, 500, 200, 10], [500, 400, 300, 200, 100, 10],
                             [512, 1024, 512, 256, 10], [512, 512, 256, 256, 10]]]

# -------------------------- DEFINE THE ATTACK SETUPS -------------------------- #
eps_attack = [0.05, 0.08]
pgd_steps_attack = [10, 50]
eot_or_collective_attack = [False, True]
eot_steps_attack = [10, 50]

# get an attack iterator
attack_it = list(it.product(eps_attack, pgd_steps_attack, eot_or_collective_attack))  # store the generator on disk for
                                                                                      # reuse

# -------------------------- RUN THE EXPERIMENTS -------------------------- #

# SIMPLE FULLY CONNECTED
# No adversarial training
if args.experiment == 'non_adv_fc':
    iter_prod_fc_not_adversarial = it.product(architectures, data_set_names)
    comb_len = len(architectures) * len(data_set_names)
    print('Fully Connected NN')
    print('Traditionally Trained')
    start = time()

    for i, arg_tup in enumerate(iter_prod_fc_not_adversarial):
        print(f'Progress: {np.around(i / comb_len * 100, 2)}%    Time elapsed: {np.around(time() - start, 1)}s',
              end='\r')
        folder_prompt = f'fc_nonadv_{arg_tup[1]}_arch{arg_tup[0]}'
        if os.path.isfile(f'pre_trained_models/fc_nets/{folder_prompt}/' + folder_prompt + '.csv') and args.no_warm_start:
            continue
        else:
            tester = FCNetTester(device=DEVICE, test_loader=data_set_test_loaders[arg_tup[1]], verbose=False)
            current_list = []
            for epoch in training_epochs:
                fc_net = torch.load(
                    f'pre_trained_models/fc_nets/{folder_prompt}/' + folder_prompt + f'_epoch{epoch}.pt').to(DEVICE)
                accuracy = tester.test(fc_net)
                for attack_params in attack_it:
                    if attack_params[2]:
                        for eot_step in eot_steps_attack:
                            attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05, EOT=True,
                                         n=eot_step)
                            adv_accuracy = tester.adversarial_test(fc_net, attack=attack)
                            curr_line = np.array(['fc', arg_tup[0], arg_tup[1], epoch, 'non_adv', 0.0, 0, False, 0,
                                                  attack_params[0], attack_params[1], True, eot_step, accuracy,
                                                  adv_accuracy], dtype='object')
                            current_list.append(curr_line)
                    else:
                        attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05)
                        adv_accuracy = tester.adversarial_test(fc_net, attack=attack)
                        curr_line = np.array(['fc', arg_tup[0], arg_tup[1], epoch, 'non_adv', 0.0, 0, False, 0,
                                              attack_params[0], attack_params[1], False, 0, accuracy, adv_accuracy],
                                             dtype='object')
                        current_list.append(curr_line)
            curr_df = pd.DataFrame(current_list, columns=header)
            curr_df.to_csv(f'pre_trained_models/fc_nets/{folder_prompt}/' + folder_prompt + '.csv')

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
        if os.path.isfile(f'pre_trained_models/fc_nets/{folder_prompt}/' + folder_prompt + '.csv') and args.no_warm_start:
            continue
        else:
            tester = FCNetTester(device=DEVICE, test_loader=data_set_test_loaders[arg_tup[1]], verbose=False)
            current_list = []
            for epoch in training_epochs:
                fc_net = torch.load(
                    f'pre_trained_models/fc_nets/{folder_prompt}/' + folder_prompt + f'_epoch{epoch}.pt').to(DEVICE)
                accuracy = tester.test(fc_net)
                for attack_params in attack_it:
                    if attack_params[2]:
                        for eot_step in eot_steps_attack:
                            attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05, EOT=True,
                                         n=eot_step)
                            adv_accuracy = tester.adversarial_test(fc_net, attack=attack)
                            curr_line = np.array(
                                ['fc', arg_tup[0], arg_tup[1], epoch, 'adv', arg_tup[2], arg_tup[3], False, 0,
                                 attack_params[0], attack_params[1], True, eot_step, accuracy,
                                 adv_accuracy], dtype='object')
                            current_list.append(curr_line)
                    else:
                        attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05)
                        adv_accuracy = tester.adversarial_test(fc_net, attack=attack)
                        curr_line = np.array(['fc', arg_tup[0], arg_tup[1], epoch, 'adv', arg_tup[2], arg_tup[3], False, 0,
                                              attack_params[0], attack_params[1], False, 0, accuracy, adv_accuracy],
                                             dtype='object')
                        current_list.append(curr_line)
            curr_df = pd.DataFrame(current_list, columns=header)
            curr_df.to_csv(f'pre_trained_models/fc_nets/{folder_prompt}/' + folder_prompt + '.csv')

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
        if os.path.isfile(f'pre_trained_models/fc_ensemble_nets/{folder_prompt}/' + folder_prompt + '.csv') and \
                args.no_warm_start:
            continue
        else:
            tester = EnsembleOfFCTester(device=DEVICE, test_loader=data_set_test_loaders[arg_tup[1]], verbose=False)
            current_list = []
            for epoch in training_epochs:
                fc_net = torch.load(
                    f'pre_trained_models/fc_ensemble_nets/{folder_prompt}/' + folder_prompt + f'_epoch{epoch}.pt').to(
                    DEVICE)
                accuracy = tester.test(fc_net)
                for attack_params in attack_it:
                    if attack_params[2]:
                        for eot_step in eot_steps_attack:
                            attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05)
                            adv_accuracy = tester.adversarial_test(fc_net, attack=attack, mode='collective')
                            curr_line = np.array(['fc_ensemble', arg_tup[0], arg_tup[1], epoch, 'non_adv', 0.0, 0, False, 0,
                                                  attack_params[0], attack_params[1], True, eot_step, accuracy,
                                                  adv_accuracy], dtype='object')
                            current_list.append(curr_line)
                    else:
                        attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05)
                        adv_accuracy = tester.adversarial_test(fc_net, attack=attack)
                        curr_line = np.array(['fc_ensemble', arg_tup[0], arg_tup[1], epoch, 'non_adv', 0.0, 0, False, 0,
                                              attack_params[0], attack_params[1], False, 0, accuracy, adv_accuracy],
                                             dtype='object')
                        current_list.append(curr_line)
            curr_df = pd.DataFrame(current_list, columns=header)
            curr_df.to_csv(f'pre_trained_models/fc_ensemble_nets/{folder_prompt}/' + folder_prompt + '.csv')


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
        if os.path.isfile(f'pre_trained_models/fc_ensemble_nets/{folder_prompt}/' + folder_prompt + '.csv') \
                and args.no_warm_start:
            continue
        else:
            eot_or_collective_training = True if args.mode == 'collective' else False
            tester = EnsembleOfFCTester(device=DEVICE, test_loader=data_set_test_loaders[arg_tup[1]], verbose=False)
            current_list = []
            for epoch in training_epochs:
                fc_net = torch.load(
                    f'pre_trained_models/fc_ensemble_nets/{folder_prompt}/' + folder_prompt + f'_epoch{epoch}.pt').to(
                    DEVICE)
                accuracy = tester.test(fc_net)
                for attack_params in attack_it:
                    if attack_params[2]:
                        for eot_step in eot_steps_attack:
                            attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05)
                            adv_accuracy = tester.adversarial_test(fc_net, attack=attack, mode='collective')
                            curr_line = np.array(
                                ['fc_ensemble', arg_tup[0], arg_tup[1], epoch, 'adv', arg_tup[2], arg_tup[3],
                                 eot_or_collective_training, 0,
                                 attack_params[0], attack_params[1], True, eot_step, accuracy,
                                 adv_accuracy], dtype='object')
                            current_list.append(curr_line)
                    else:
                        attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05)
                        adv_accuracy = tester.adversarial_test(fc_net, attack=attack, mode='individual')
                        curr_line = np.array(['fc_ensemble', arg_tup[0], arg_tup[1], epoch, 'adv', arg_tup[2], arg_tup[3],
                                              eot_or_collective_training, 0,
                                              attack_params[0], attack_params[1], False, 0, accuracy, adv_accuracy],
                                             dtype='object')
                        current_list.append(curr_line)
            curr_df = pd.DataFrame(current_list, columns=header)
            curr_df.to_csv(f'pre_trained_models/fc_ensemble_nets/{folder_prompt}/' + folder_prompt + '.csv')

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
        if os.path.isfile(f'pre_trained_models/rse_nets/{folder_prompt}/' + folder_prompt + '.csv') and args.no_warm_start:
            continue
        else:
            tester = FullyConnectedRSETester(device=DEVICE, test_loader=data_set_test_loaders[arg_tup[1]], verbose=False)
            current_list = []
            for epoch in training_epochs:
                rse_net = torch.load(
                    f'pre_trained_models/rse_nets/{folder_prompt}/' + folder_prompt + f'_epoch{epoch}.pt').to(DEVICE)
                accuracy = tester.test(rse_net)
                for attack_params in attack_it:
                    if attack_params[2]:
                        for eot_step in eot_steps_attack:
                            attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05, EOT=True,
                                         n=eot_step)
                            adv_accuracy = tester.adversarial_test(rse_net, attack=attack)
                            curr_line = np.array(['rse', arg_tup[0], arg_tup[1], epoch, 'non_adv', 0.0, 0, False, 0,
                                                  attack_params[0], attack_params[1], True, eot_step, accuracy,
                                                  adv_accuracy], dtype='object')
                            current_list.append(curr_line)
                    else:
                        attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05)
                        adv_accuracy = tester.adversarial_test(rse_net, attack=attack)
                        curr_line = np.array(['rse', arg_tup[0], arg_tup[1], epoch, 'non_adv', 0.0, 0, False, 0,
                                              attack_params[0], attack_params[1], False, 0, accuracy, adv_accuracy],
                                             dtype='object')
                        current_list.append(curr_line)
            curr_df = pd.DataFrame(current_list, columns=header)
            curr_df.to_csv(f'pre_trained_models/rse_nets/{folder_prompt}/' + folder_prompt + '.csv')

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
        if os.path.isfile(f'pre_trained_models/rse_nets/{folder_prompt}/' + folder_prompt + '.csv') and args.no_warm_start:
            continue
        else:
            eot_or_collective_training = args.eot
            tester = FullyConnectedRSETester(device=DEVICE, test_loader=data_set_test_loaders[arg_tup[1]], verbose=False)
            current_list = []
            for epoch in training_epochs:
                rse_net = torch.load(
                    f'pre_trained_models/rse_nets/{folder_prompt}/' + folder_prompt + f'_epoch{epoch}.pt').to(
                    DEVICE)
                accuracy = tester.test(rse_net)
                for attack_params in attack_it:
                    if attack_params[2]:
                        for eot_step in eot_steps_attack:
                            attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05, EOT=True,
                                         n=eot_step)
                            adv_accuracy = tester.adversarial_test(rse_net, attack=attack)
                            curr_line = np.array(
                                ['rse', arg_tup[0], arg_tup[1], epoch, 'adv', arg_tup[2], arg_tup[3],
                                 eot_or_collective_training, 0,
                                 attack_params[0], attack_params[1], True, eot_step, accuracy,
                                 adv_accuracy], dtype='object')
                            current_list.append(curr_line)
                    else:
                        attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05)
                        adv_accuracy = tester.adversarial_test(rse_net, attack=attack)
                        curr_line = np.array(['rse', arg_tup[0], arg_tup[1], epoch, 'adv', arg_tup[2], arg_tup[3],
                                              eot_or_collective_training, 0,
                                              attack_params[0], attack_params[1], False, 0, accuracy, adv_accuracy],
                                             dtype='object')
                        current_list.append(curr_line)
            curr_df = pd.DataFrame(current_list, columns=header)
            curr_df.to_csv(f'pre_trained_models/rse_nets/{folder_prompt}/' + folder_prompt + '.csv')

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
        if os.path.isfile(f'pre_trained_models/bnn_nets/{folder_prompt}/' + folder_prompt + '.csv') and args.no_warm_start:
            continue
        else:
            tester = FullyConnectedBNNTester(device=DEVICE, test_loader=data_set_test_loaders[arg_tup[1]], verbose=False)
            current_list = []
            for epoch in training_epochs:
                bnn_net = torch.load(
                    f'pre_trained_models/bnn_nets/{folder_prompt}/' + folder_prompt + f'_epoch{epoch}.pt').to(DEVICE)
                accuracy = tester.test(bnn_net)
                for attack_params in attack_it:
                    if attack_params[2]:
                        for eot_step in eot_steps_attack:
                            attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05, EOT=True,
                                         n=eot_step)
                            adv_accuracy = tester.adversarial_test(bnn_net, attack=attack)
                            curr_line = np.array(['bnn', arg_tup[0], arg_tup[1], epoch, 'non_adv', 0.0, 0, False, 0,
                                                  attack_params[0], attack_params[1], True, eot_step, accuracy,
                                                  adv_accuracy], dtype='object')
                            current_list.append(curr_line)
                    else:
                        attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05)
                        adv_accuracy = tester.adversarial_test(bnn_net, attack=attack)
                        curr_line = np.array(['bnn', arg_tup[0], arg_tup[1], epoch, 'non_adv', 0.0, 0, False, 0,
                                              attack_params[0], attack_params[1], False, 0, accuracy, adv_accuracy],
                                             dtype='object')
                        current_list.append(curr_line)
            curr_df = pd.DataFrame(current_list, columns=header)
            curr_df.to_csv(f'pre_trained_models/bnn_nets/{folder_prompt}/' + folder_prompt + '.csv')

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
        if os.path.isfile(f'pre_trained_models/bnn_nets/{folder_prompt}/' + folder_prompt + '.csv') and args.no_warm_start:
            continue
        else:
            eot_or_collective_training = args.eot
            tester = FullyConnectedBNNTester(device=DEVICE, test_loader=data_set_test_loaders[arg_tup[1]], verbose=False)
            current_list = []
            for epoch in training_epochs:
                bnn_net = torch.load(
                    f'pre_trained_models/bnn_nets/{folder_prompt}/' + folder_prompt + f'_epoch{epoch}.pt').to(
                    DEVICE)
                accuracy = tester.test(bnn_net)
                for attack_params in attack_it:
                    if attack_params[2]:
                        for eot_step in eot_steps_attack:
                            attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05, EOT=True,
                                         n=eot_step)
                            adv_accuracy = tester.adversarial_test(bnn_net, attack=attack)
                            curr_line = np.array(
                                ['bnn', arg_tup[0], arg_tup[1], epoch, 'adv', arg_tup[2], arg_tup[3],
                                 eot_or_collective_training, 0,
                                 attack_params[0], attack_params[1], True, eot_step, accuracy,
                                 adv_accuracy], dtype='object')
                            current_list.append(curr_line)
                    else:
                        attack = PGD(epsilon=attack_params[0], n_iterations=attack_params[1], step_size=0.05)
                        adv_accuracy = tester.adversarial_test(bnn_net, attack=attack)
                        curr_line = np.array(['bnn', arg_tup[0], arg_tup[1], epoch, 'adv', arg_tup[2], arg_tup[3],
                                              eot_or_collective_training, 0,
                                              attack_params[0], attack_params[1], False, 0, accuracy, adv_accuracy],
                                             dtype='object')
                        current_list.append(curr_line)
            curr_df = pd.DataFrame(current_list, columns=header)
            curr_df.to_csv(f'pre_trained_models/bnn_nets/{folder_prompt}/' + folder_prompt + '.csv')

else:
    raise RuntimeError('If you read this message then something must have gone horribly wrong')
