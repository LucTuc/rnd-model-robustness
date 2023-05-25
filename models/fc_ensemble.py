from .fc_net import FullyConnected, FCNetTrainer, FCNetTester
import torch
import torch.nn as nn
import numpy as np


class EnsembleOfFullyConnected(nn.Module):
    """
    An output averaged ensemble of fully connected neural networks. Builds upon the 'FullyConnected' class.
    """
    def __init__(self, device, input_size, member_layouts, data_mean, data_std):
        """
        Constructor of the ensemble.

        :param device: (str) Name of the device on which the intermediate tensors are to be stored.
        :param input_size: (int) One of the spatial dimensions of a one channel square input image. The resulting input
            layer will be of input_size * input_size neurons.
        :param member_layouts: (list of lists) A list of all ensemble member's layout where each layout describes a
            fully connected network. In each of the layout lists, the first l-1 entries correspond to the hidden layers
            followed by a ReLU activation and the last l-th entry specifies the number of output classes.
        """
        super(EnsembleOfFullyConnected, self).__init__()
        self.member_layouts = member_layouts
        self.device = device
        self.members = [FullyConnected(device, input_size, layout, data_mean, data_std).to(device) for layout in member_layouts]

    def reset_parameters(self):
        """
        Resets the parameters of the ensemble.

        :return: self
        """
        for member in self.members:
            member.reset_parameters()
        return self

    def forward(self, x, return_std=False):
        """
        Modified forward method of the ensemble. It returns the average class probabilities over the networks, and
        optionally the standard deviation as an estimate for the uncertainty.

        :param x: (torch.Tensor) The batch of input data to push through the ensemble (sequentially through each member
            network).
        :param return_std: (bool) Set to True if you wish to return the standard deviation of the output probabilities.
            The default is False to comply with the 'nn.Module' base class forward method.
        :return: (torch.Tensor or tuple of torch.Tensors) The mean output probabilities of each member network, and if
            'return_std=True' additionally it returns the standard deviations for each of the probabilities as well.
        """
        member_outputs = torch.stack([member.forward(x) for member in self.members], dim=0)
        if return_std:
            return torch.mean(member_outputs, dim=0), torch.std(member_outputs, dim=0)
        else:
            return torch.mean(member_outputs, dim=0)


class EnsembleOfFCTrainer:
    """
    A wrapper class to train the ensemble.
    """
    def __init__(self, train_loader, n_epochs, optimizers, criterion, device, verbose=True):
        """
        The trainer constructor.

        :param train_loader: (DataLoader) The torch DataLoader object containing the transformed training data.
        :param n_epochs: (int) The desired number of training epochs.
        :param optimizers: (list of Optimizer) The instantiated torch Optimizer objects chosen for the training of each
            member network.
        :param criterion: (_WeightedLoss) The instantiated loss function.
        :param device: (str) The name of the device on which the intermediate tensors are to be stored.
        :param verbose: (bool) Toggle if you want to print the progress.
        """
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.optimizers = optimizers
        self.criterion = criterion
        self.device = device
        self.verbose = verbose

    def train(self, ensemble_net, checkpoints=None, save_path_base=None):
        """
        Traditional training of the ensemble model by training each member network individually hence introducing
        unaccounted noise. Make sure that each of the models are on the same device as the one contained in the
        constructor.

        :param ensemble_net: (nn.Module) The ensemble to be trained.
        :param checkpoints: (list) List of epochs at which the model shall be saved.
        :param save_path_base: (str) The path to which the model shall be saved if a list of checkpoints is given.
        :return: None
        """
        if checkpoints is None:
            checkpoints = []
        for epoch in range(self.n_epochs):
            for member, optimizer in zip(ensemble_net.members, self.optimizers):
                running_loss = []
                for i, data in enumerate(self.train_loader, 0):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)

                    optimizer.zero_grad()
                    outputs = member(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += [loss.item()]
                    if i % 100 == 99 and self.verbose:
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)), end='\r')
                        running_loss = []
                if self.verbose:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)))
            if (epoch + 1) in checkpoints:
                torch.save(ensemble_net, save_path_base + f'_epoch{epoch + 1}.pt')
        if self.verbose:
            print('Finished Training')

    def adversarial_train(self, ensemble_net, attack, mode='individual', checkpoints=None, save_path_base=None):
        """
        Adversarial training of an ensemble of neural networks on a given attack. The training can have two modes:
            1. 'individual': each member of the ensemble is individually adversarially trained.
            2. 'collective': the ensembles is collectively adversarially trained, using the gradient through the
                averaged output.

        :param ensemble_net: (nn.Module) The ensemble to be trained.
        :param attack: The desired adversarial attack class that implements a '.attack(net)' method.
        :param mode: (str) The adversarial training mode. Available modes are:
            1. 'individual': each member of the ensemble is individually adversarially trained.
            2. 'collective': the ensembles is collectively adversarially trained, using the gradient through the
                averaged output.
        :param checkpoints: (list) List of epochs at which the model shall be saved.
        :param save_path_base: (str) The path to which the model shall be saved if a list of checkpoints is given.
        :return: None
        """
        if checkpoints is None:
            checkpoints = []
        if mode == 'individual':
            for epoch in range(self.n_epochs):
                for member, optimizer in zip(ensemble_net.members, self.optimizers):
                    running_loss = []
                    for i, data in enumerate(self.train_loader, 0):
                        original_inputs, labels = data[0].to(self.device), data[1].to(self.device)
                        adversarial_inputs = attack.attack(member, original_inputs, labels)
                        optimizer.zero_grad()
                        outputs = member(adversarial_inputs)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += [loss.item()]
                        if i % 100 == 99 and self.verbose:
                            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)), end='\r')
                            running_loss = []
                    if self.verbose:
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)))
                if (epoch + 1) in checkpoints:
                    torch.save(ensemble_net, save_path_base + f'_epoch{epoch + 1}.pt')
            if self.verbose:
                print('Finished Training')

        if mode == 'collective':
            for epoch in range(self.n_epochs):
                running_loss = []
                for i, data in enumerate(self.train_loader, 0):
                    original_inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    adversarial_inputs = attack.attack(ensemble_net, original_inputs, labels)
                    self.optimizers[0].zero_grad()
                    outputs = ensemble_net.forward(adversarial_inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizers[0].step()

                    running_loss += [loss.item()]
                    if i % 100 == 99 and self.verbose:
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, np.mean(running_loss)), end='\r')
                        running_loss = []
                if self.verbose:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)))
                if (epoch + 1) in checkpoints:
                    torch.save(ensemble_net, save_path_base + f'_epoch{epoch + 1}.pt')
            if self.verbose:
                print('Finished Training')
        else:
            return None


class EnsembleOfFCTester:
    """
    A wrapper class to test ensembles of neural networks.
    """
    def __init__(self, test_loader, device, verbose=True):
        """
        Constructor of the tester.

        :param test_loader: (DataLoader) The torch DataLoader containing the transformed test data.
        :param device: (str) The name of the device on which the intermediate tensors are to be stored.
        :param verbose: (bool) Control the prints.
        """
        self.test_loader = test_loader
        self.device = device
        self.verbose = verbose

    def test(self, ensemble_net):
        """
        Calculates the test accuracy of the given ensemble network as the ratio between the correctly predicted labels
        and all labels.

        :param ensemble_net: (nn.Module) The ensemble to be tested.
        :return: (float) The accuracy as the ratio of the correctly predicted labels and all labels.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                means = ensemble_net.forward(images)
                _, predicted = torch.max(means.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if self.verbose:
            print(f'Accuracy of the network on the {total} test images: %d %%' % (100 * correct / total))

        return correct/total

    def adversarial_test(self, ensemble_net, attack, mode='individual'):
        """
        Calculates the robust accuracy of the ensemble network on a given attack and for a given mode. The available
        modes are:
            1. 'individual': For each sample we select one member of the ensemble at random and generate the adversarial
                sample by attacking that one. Then we observe the prediction on the ensemble.
            2. 'collective': We attack the ensemble through the averaged output.

        :param ensemble_net: (nn.Module) The ensemble network for which the robust accuracy is to be calculated.
        :param attack: The attack we calculate the robust accuracy against.
        :param mode: (str) The mode of the attack. There are two modes available:
            1. 'individual': For each sample we select one member of the ensemble at random and generate the adversarial
                sample by attacking that one. Then we observe the prediction on the ensemble.
            2. 'collective': We attack the ensemble through the averaged output.
        :return: (float) Robust accuracy as the average of 1-(relative frequency of successful attacks) for each class.
        """
        if mode == 'individual':
            total = 0
            correct = 0

            for data in self.test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                for member in ensemble_net.members:
                    adv_images = attack.attack(member, images, labels)
                    means = ensemble_net.forward(adv_images)
                    _, predicted = torch.max(means.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            if self.verbose:
                print(f'Adverserial Accuracy of the network on the {int(total / len(ensemble_net.members))} '
                      f'test images: %d %%' % (100 * correct / total))

            return correct/total

        elif mode == 'collective':
            correct = 0
            total = 0
            for data in self.test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                adv_images = attack.attack(ensemble_net, images, labels)
                means = ensemble_net.forward(adv_images).detach()
                _, predicted = torch.max(means.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            if self.verbose:
                print(f'Adverserial Accuracy of the network on the {total} test images: %d %%' % (100 * correct / total))

            return correct/total
        else:
            raise NotImplementedError
