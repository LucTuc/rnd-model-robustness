import torch
import torch.nn as nn
import numpy as np


class Normalization(nn.Module):
    """
    Normalization layer. This allows us to create adversarial examples with respect to the actual
    unnormalized input without difficulties.
    """
    def __init__(self, device, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([mean]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([std]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class LinReLU(nn.Module):
    """
    A linear layer followed by a ReLU activation layer.
    """
    def __init__(self, in_size, out_size):
        super(LinReLU, self).__init__()

        self.Linear = nn.Linear(in_size, out_size)
        self.ReLU = nn.ReLU()

    def reset_parameters(self):
        self.Linear.reset_parameters()
        return self

    def forward(self, x):
        x = self.Linear(x)
        return self.ReLU(x)


class FullyConnected(nn.Module):
    """
    A fully connected neural network with ReLU activations.
    """
    def __init__(self, device, input_size, fc_layers, data_mean, data_std):
        """
        The constructor of the fully connected neural network.

        :param device: (str) The name of the device on which the intermediate tensors are to be stored.
        :param input_size: (int) One of the spatial dimensions of a one channel square input image. The resulting input
            layer will be of input_size * input_size neurons.
        :param fc_layers: (list) A list specifying the architecture of the fully connected network. The first l-1
            entries correspond to the hidden layers followed by a ReLU activation and the last l-th entry specifies the
            number of output classes.
        """
        super(FullyConnected, self).__init__()

        layers = [Normalization(device, data_mean, data_std), nn.Flatten()]
        prev_fc_size = input_size**2
        for i, fc_size in enumerate(fc_layers):
            if i + 1 < len(fc_layers):
                layers += [LinReLU(prev_fc_size, fc_size)]
            else:
                layers += [nn.Linear(prev_fc_size, fc_size)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def reset_parameters(self):
        """
        Reset the parameters of the network.

        :return: self
        """
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        return self

    def forward(self, x):
        x = self.layers(x)
        return x
        # return nn.functional.softmax(x, dim=1)  # returns probabilities


# heavily building upon: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class FCNetTrainer:
    """
    A wrapper class for training a fully connected neural network. Includes the option of traditional training and
    adversarial training.
    """
    def __init__(self, train_loader, n_epochs, optimizer, criterion, device, verbose=True):
        """
        The constructor of the fully connected network trainer.

        :param train_loader: (DataLoader) The torch DataLoader object containing the transformed training data.
        :param n_epochs: (int) The desired number of training epochs.
        :param optimizer: (Optimizer) The instantiated torch Optimizer object chosen for the training.
        :param criterion: (_WeightedLoss) The instantiated loss function.
        :param device: (str) The name of the device on which the intermediate tensors are to be stored.
        :param verbose: (bool) Toggle if you want to print the progress.
        """
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.verbose = verbose

    def train(self, net, checkpoints=None, save_path_base=None):
        """
        Traditional training method. The training progress is printed each 100 batches. No intermediate validation.

        :param net: (nn.Module) The pytorch network to be trained. Make sure that the network is already on the target
            device.
        :param checkpoints: (list) List of epochs at which the model shall be saved.
        :param save_path_base: (str) The path to which the model shall be saved if a list of checkpoints is given.
        :return: None
        """
        if checkpoints is None:
            checkpoints = []
        for epoch in range(self.n_epochs):
            running_loss = []
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += [loss.item()]
                if i % 100 == 99 and self.verbose:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)), end='\r')
                    running_loss = []
            if self.verbose:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)))
            if (epoch + 1) in checkpoints:
                torch.save(net, save_path_base + f'_epoch{epoch + 1}.pt')
        if self.verbose:
            print('Finished Training')

    def adversarial_train(self, net, attack, checkpoints=None, save_path_base=None):
        """
        Adversarial training method where the training adversarial samples are generated by a given attack.

        :param net: (nn.Module) The pytorch network to be trained. Make sure that the network is already on the target
            device.
        :param attack: The desired adversarial attack class that implements a '.attack(net, x, y)' method.
        :param checkpoints: (list) List of epochs at which the model shall be saved.
        :param save_path_base: (str) The path to which the model shall be saved if a list of checkpoints is given.
        :return: None
        """
        if checkpoints is None:
            checkpoints = []
        for epoch in range(self.n_epochs):
            running_loss = []
            for i, data in enumerate(self.train_loader, 0):
                original_inputs, labels = data[0].to(self.device), data[1].to(self.device)
                adversarial_inputs = attack.attack(net, original_inputs, labels)  # adversarial example included
                self.optimizer.zero_grad()
                outputs = net(adversarial_inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += [loss.item()]
                if i % 100 == 99 and self.verbose:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)), end='\r')
                    running_loss = []
            if self.verbose:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)))
            if (epoch + 1) in checkpoints:
                torch.save(net, save_path_base + f'_epoch{epoch+1}.pt')
        if self.verbose:
            print('Finished Training')


# heavily building upon: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class FCNetTester:
    """
    A wrapper class for testing trained fully connected neural networks.
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

    def test(self, net):
        """
        Calculates the accuracy of the given network on the test data set.

        :param net: (nn.Module) The torch model to be tested.
        :return: (float) Accuracy as share of correctly predicted labels from all labels.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)  # take the maximal output
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if self.verbose:
            print(f'Accuracy of the network on the {total} test images: %d %%' % (100 * correct / total))

        return correct/total

    def adversarial_test(self, net, attack):
        """
        Calculates the robust accuracy of a fully connected neural network when attacked by a given attack.

        :param net: (nn.Module) The torch model to be tested.
        :param attack: The desired adversarial attack class that implements a '.attack(net)' method.
        :return: (float) Robust accuracy as the average of 1-(relative frequency of successful attacks) for each class.
        """
        correct = 0
        total = 0
        for data in self.test_loader:
            images, labels = data[0].to(self.device), data[1].to(self.device)
            adv_image = attack.attack(net, images, labels)
            outputs = net(adv_image).detach()
            _, predicted = torch.max(outputs.data, 1)  # take the maximal output
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if self.verbose:
            print(f'Adverserial Accuracy of the network on the {total} test images: %d %%' % (100 * correct / total))

        return correct/total
