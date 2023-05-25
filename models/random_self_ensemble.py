import torch
import torch.nn as nn
import numpy as np


class Noise(nn.Module):
    # License Included, original source code: https://github.com/xuanqing94/RobustNet
    """
    MIT License

    Copyright (c) 2017 Xuanqing Liu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self, std, device):
        super(Noise, self, ).__init__()
        self.std = std
        self.device = device
        self.buffer = None

    def forward(self, x):
        if self.buffer is None:
            self.buffer = torch.Tensor(x.size()).normal_(0, self.std).to(self.device)
        else:
            self.buffer.resize_(x.size()).normal_(0, self.std)
        x.data += self.buffer
        return x


class Normalization(nn.Module):
    """
    Normalization layer hard coded for MNIST. This allows us to create adversarial examples with respect to the actual
    unnormalized input without difficulties.
    """
    def __init__(self, device, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([mean]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([std]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class NoiseLinReLU(nn.Module):
    """
    Layer combination of a noisy layer, then a linear layer, and a ReLU activation layer.
    """
    def __init__(self, in_size, out_size, std, device):
        super(NoiseLinReLU, self).__init__()

        self.Noise = Noise(std=std, device=device)
        self.Linear = nn.Linear(in_size, out_size)
        self.ReLU = nn.ReLU()

    def reset_parameters(self):
        self.Linear.reset_parameters()
        return self

    def forward(self, x):
        x1 = self.Noise(x)
        x = self.Linear(x1)
        return self.ReLU(x)


class FullyConnectedRandomSelfEnsemble(nn.Module):
    """
    A fully connected Random Self Ensemble (RSE) on the basis of: Liu et al. 2017;
    online: https://arxiv.org/abs/1712.00673.
    """
    def __init__(self, device, input_size, fc_layers, std_init, std_inner, data_mean, data_std):
        """
        Constructor of the Random Self Ensemble.

        :param device: (str) The name of the device on which the intermediate tensors are to be stored.
        :param input_size: (int) One of the spatial dimensions of a one channel square input image. The resulting input
            layer will be of input_size * input_size neurons.
        :param fc_layers: (list) A list specifying the architecture of the fully connected network. The first l-1
            entries correspond to the hidden layers followed by a ReLU activation and the last l-th entry specifies the
            number of output classes.
        :param std_init: (float) The variance of the Gaussian noise added to the input to the first layer. TODO: or std?
        :param std_inner: (float) The variance of the Gaussian noise added to the input to the rest of the layers.
            TODO: or std?
        """
        super(FullyConnectedRandomSelfEnsemble, self).__init__()
        layers = [Normalization(device, data_mean, data_std), nn.Flatten()]
        prev_fc_size = input_size**2
        for i, fc_size in enumerate(fc_layers):
            if i == 0:
                layers += [NoiseLinReLU(prev_fc_size, fc_size, std=std_init, device=device)]
            elif i + 1 < len(fc_layers):
                layers += [NoiseLinReLU(prev_fc_size, fc_size, std=std_inner, device=device)]
            else:
                layers += [Noise(std=std_inner, device=device), nn.Linear(prev_fc_size, fc_size)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def reset_parameters(self):
        """
        Reset the parameters of the RSE.

        :return: self
        """
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        return self

    def forward(self, x, n=1, return_std=False):
        """
        Modified forward function of the RSE. Does 'n' forward passes through the network where each time a different
        noise term is sampled. To obtain the output of the RSE we average over the sampled outputs.

        :param x: (torch.Tensor) The batch of input data to push through the RSE.
        :param n: (int) The number of passes through the network to average up. The argument is defaulted (to 10) to
            comply with the nn.Module descriptor class' use of the forward method.
        :param return_std: (bool) Toggle to return an uncertainty estimate over the predicted probabilities in the form
            of the samples' standard deviation. The argument is defaulted (to False) to comply with the nn.Module
            descriptor class' use of the forward method. Note that this is different to the std arguments given to the
            model.
        :return: (torch.Tensor or tuple of torch.Tensors) The mean output probabilities of each member network, and if
            'return_std=True' additionally it returns the standard deviations for each of the probabilities as well.
        """
        outputs = []
        for _ in range(n):
            # interim = self.layers(x)
            # outputs += [nn.functional.softmax(interim, dim=1)]
            outputs += [self.layers(x)]
        outputs = torch.stack(outputs, dim=0)
        if return_std:
            return torch.mean(outputs, dim=0), torch.std(outputs, dim=0)
        else:
            return torch.mean(outputs, dim=0)


class FullyConnectedRSETrainer:
    """
    A wrapper class to train fully connected RSEs.
    """
    def __init__(self, train_loader, n_epochs, optimizer, criterion, device, verbose=True):
        """
        RSE trainer class constructor.

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

    def train(self, rse_net, checkpoints=None, save_path_base=None):
        """
        Train the given RSE on the training data set. Prints the progress at every 100th batch.

        :param rse_net: (nn.Module) The RSE to be trained.
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
                outputs = rse_net.forward(inputs, 1)
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
                torch.save(rse_net, save_path_base + f'_epoch{epoch + 1}.pt')
        if self.verbose:
            print('Finished Training')

    def adversarial_train(self, rse_net, attack, checkpoints=None, save_path_base=None):
        """
        Adversarial training of the RSE on a given attack.

        :param rse_net: (nn.Module) The RSE to be trained.
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
                adversarial_inputs = attack.attack(rse_net, original_inputs, labels)  # adversarial example included

                self.optimizer.zero_grad()
                outputs = rse_net.forward(adversarial_inputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += [loss.item()]
                if i % 100 == 99 and self.verbose:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, np.mean(running_loss)), end='\r')
                    running_loss = []
            if self.verbose:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)))
            if (epoch + 1) in checkpoints:
                torch.save(rse_net, save_path_base + f'_epoch{epoch + 1}.pt')
        if self.verbose:
            print('Finished Training')


class FullyConnectedRSETester:
    """
    A wrapper class to test RSEs.
    """
    def __init__(self, test_loader, device, verbose):
        """
        RSE tester constructor.

        :param test_loader: (DataLoader) The torch DataLoader containing the transformed test data.
        :param device: (str) The name of the device on which the intermediate tensors are to be stored.
        :param verbose: (bool) Control the prints.
        """
        self.test_loader = test_loader
        self.device = device
        self.verbose = verbose

    def test(self, rse_net, n=10):
        """
        Calculates the test accuracy of the RSE over 'n' passes for the same batch of data.

        :param rse_net: (nn.Module) The RSE to be tested.
        :param n: (int) The number of forward passes through the network at different noise to obtain the samples for
            the calculation of the mean output. For more details see the documentation of the .forward() method of the
            class FullyConnectedRandomSelfEnsemble.
        :return: (float) Test accuracy as the ratio between the correctly predicted labels and all labels.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs, stds = rse_net.forward(images, n, return_std=True)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # for output, std, pred, label in zip(outputs, stds, predicted, labels):
                #     if pred != label:
                #         print(f'Predicted: {pred} Label: {label}')
                #         print(output, std)
                correct += (predicted == labels).sum().item()

        if self.verbose:
            print(f'Accuracy of the network on the {total} test images: %d %%' % (100 * correct / total))

        return correct/total

    def adversarial_test(self, rse_net, attack, n=10):
        """
        Calculates the robust accuracy for an RSE on a given attack.

        :param rse_net: (nn.Module) The RSE to be tested.
        :param attack: The desired adversarial attack class that implements a '.attack(net)' method.
        :param n: (int) The number of forward passes through the network at different noise to obtain the samples for
            the calculation of the mean output. For more details see the documentation of the .forward() method of the
            class FullyConnectedRandomSelfEnsemble.
        :return: (float) Robust accuracy as the average of 1-(relative frequency of successful attacks) for each class.
        """
        correct = 0
        total = 0
        for data in self.test_loader:
            images, labels = data[0].to(self.device), data[1].to(self.device)
            adv_image = attack.attack(rse_net, images, labels)
            outputs = rse_net.forward(adv_image, n).detach()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if self.verbose:
            print(f'Accuracy of the network on the {total} test images: %d %%' % (100 * correct / total))

        return correct/total
