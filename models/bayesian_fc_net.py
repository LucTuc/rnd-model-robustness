import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math


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


class BayesianLinear(nn.Module):
    """
    A variational Bayesian linear layer.
    """
    def __init__(self, in_size, out_size, device, bias=True):
        """
        Bayesian linear layer constructor.

        :param in_size: (int) Number of input features.
        :param out_size: (int) Number of output features.
        :param device: (str) The name of the device to store the intermediate tensors on.
        :param bias: (bool) Toggle to use a bias term on the affine transformation.
        """
        super(BayesianLinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.use_bias = bias
        self.device = device
        self.weight_prior_mu = (torch.Tensor(out_size, in_size).to(device)).data.fill_(0.0)
        self.weight_prior_sigma = (torch.Tensor(out_size, in_size).to(device)).data.fill_(0.1)
        self.weight_mu = nn.Parameter(torch.Tensor(out_size, in_size).to(device))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_size, in_size).to(device))
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_size).to(device))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_size).to(device))
            self.bias_prior_mu = (torch.Tensor(out_size).to(device)).data.fill_(0.0)
            self.bias_prior_sigma = (torch.Tensor(out_size).to(device)).data.fill_(0.1)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the layer.

        :return: self
        """
        dev = 1. / math.sqrt(self.weight_mu.size()[1])  # like in AdvBNN
        self.weight_mu.data.uniform_(-dev, dev)
        self.weight_log_sigma.data.fill_(-10)  # -2.3
        if self.use_bias:
            self.bias_mu.data.uniform_(-dev, dev)
            self.bias_log_sigma.data.fill_(-10)  # -2.3
        return self

    def forward(self, inputs):
        """
        Bayesian forward method using the reparametrization trick: if x ~ N(mu, sigma), then x = mu + sigma * epsilon,
        where epsilon ~ N(0, 1).

        :param inputs: (torch.Tensor) The batch of input data to push through the layer.
        :return: (torch.Tensor) The predicted outputs of the final layer for a single random pass through the network.
        """
        # reparametrization trick
        epsilon = torch.randn((self.out_size, self.in_size)).to(self.device)
        weights = self.weight_mu + epsilon * torch.exp(self.weight_log_sigma)
        bias = None
        if self.use_bias:
            epsilon = torch.randn(self.out_size).to(self.device)
            bias = self.bias_mu + epsilon * torch.exp(self.bias_log_sigma)
        return F.linear(inputs, weights, bias)

    def kl_loss(self, beta=.01):
        """
        Calculates the KL-Divergence between the Gaussian priors of the layer and the Gaussian distributions of the
        parameters. Necessary for the ELBO loss for variational training (Bayes by Backprop). As this term effectively
        serves as a regularizer, we can control the strength of the regularization with the beta parameter.

        :param beta: (float) Sets the strength of the regularization effect coming from the kl_loss.
        :return: (torch.Tensor) The sum of the KL-Divergence between the priors and the weight distributions multiplied
            by the regularization strength parameter beta.
        """
        kl_loss = torch.log(self.weight_prior_sigma) - self.weight_log_sigma + 0.5 * \
                  (self.weight_log_sigma**2 + (self.weight_mu - self.weight_prior_mu)**2) / \
                  (self.weight_prior_sigma ** 2) - 0.5
        kl_loss = torch.mean(kl_loss)
        if self.use_bias:
            kl_loss_bias = torch.log(self.bias_prior_sigma) - self.bias_log_sigma + 0.5 * \
                           (self.bias_log_sigma**2 + (self.bias_mu - self.bias_prior_mu)**2) / \
                           (self.bias_prior_sigma**2) - 0.5
            kl_loss += torch.mean(kl_loss_bias)

        return beta * kl_loss


class BayesianFullyConnected(nn.Module):
    """
    Fully connected Bayesian Neural Network (BNN).
    """
    def __init__(self, device, input_size, fc_layers, data_mean, data_std):
        """
        BNN constructor.

        :param device: (str) Name of the device on which we store the intermediate tensors.
        :param input_size: (int) One of the spatial dimensions of a one channel square input image. The resulting input
            layer will be of input_size * input_size neurons.
        :param fc_layers: (list) A list specifying the architecture of the fully connected network. The first l-1
            entries correspond to the hidden layers followed by a ReLU activation and the last l-th entry specifies the
            number of output classes.
        """
        super(BayesianFullyConnected, self).__init__()

        layers = [Normalization(device, data_mean, data_std), nn.Flatten()]
        prev_fc_size = input_size**2
        for i, fc_size in enumerate(fc_layers):
            if i + 1 < len(fc_layers):
                layers += [BayesianLinear(prev_fc_size, fc_size, device), nn.ReLU()]
            else:
                layers += [BayesianLinear(prev_fc_size, fc_size, device)]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)
        self.device = device

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
        """
        Forward pass for a single sample of weights. Implementing the nn.Module.forward() method.

        :param x: (torch.Tensor) The batch of input data to be pushed through the network.
        :return: (torch.Tensor) The batch of output after a single forward pass.
        """
        return self.layers(x)

    def predict_class_probabilities(self, x, n=10, return_std=False):
        """
        Calculates the n-sample Monte Carlo sampled probabilities for each class. If 'return_std' is set to 'True' it
        returns also the standard deviation of the samples for each class.

        :param x: (torch.Tensor) The batch of input data to be pushed through the network n times.
        :param n: (int) Number of times we resample the weights and do a forward pass to obtain the mean output of the
            network.
        :param return_std: (bool) Toggle to return only the mean of the predicted probabilities or also the standard
            deviation over the sampled probabilities for each class.
        :return: (torch.Tensor or tuple of torch.Tensors) The mean output probabilities of each member network, and if
            'return_std=True' additionally it returns the standard deviations for each of the probabilities as well.
        """
        outputs = []
        for _ in range(n):
            interim = self.layers(x)
            outputs += [F.softmax(interim, dim=1)]
        outputs = torch.stack(outputs, dim=0)
        if return_std:
            return torch.mean(outputs, dim=0), torch.std(outputs, dim=0)
        else:
            return torch.mean(outputs, dim=0)

    def kl_loss(self):
        """
        Calculates the KL-Divergence between all the weight distributions and the weight priors cumulatively over each
        layer for the whole network.

        :return: (torch.Tensor) The mean adjusted (see the documentation of BayesianLinear) KL-Divergence in each layer.
        """
        kl = torch.Tensor([0]).to(self.device)
        for layer in self.layers:
            if 'Bayesian' in str(layer):
                kl += layer.kl_loss()
        return torch.mean(kl)


class FullyConnectedBNNTrainer:
    """
    Wrapper class to train fully connected BNNs.
    """
    def __init__(self, train_loader, n_epochs, optimizer, device, verbose=True):
        """
        Constructor of the training class.

        :param train_loader: (DataLoader) The torch DataLoader object containing the transformed training data.
        :param n_epochs: (int) The desired number of training epochs.
        :param optimizer: (Optimizer) The instantiated torch Optimizer object chosen for the training.
        :param device: (str) The name of the device on which the intermediate tensors are to be stored.
        :param verbose: (bool) Toggle if you want to print the progress.
        """
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose

    def train(self, bnn_net, checkpoints=None, save_path_base=None):
        """
        Trains the BNN on the training data set using variational bayes (Bayes by Backprop).

        :param bnn_net: (nn.Module) The torch BNN model.
        :param checkpoints: (list) List of epochs at which the model shall be saved.
        :param save_path_base: (str) The path to which the model shall be saved if a list of checkpoints is given.
        :return: None
        """
        if checkpoints is None:
            checkpoints = []
        cross_entropy = nn.CrossEntropyLoss()
        for epoch in range(self.n_epochs):
            running_loss = []
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()
                outputs = bnn_net(inputs)
                loss = cross_entropy(outputs, labels) + bnn_net.kl_loss()
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
                torch.save(bnn_net, save_path_base + f'_epoch{epoch + 1}.pt')
        if self.verbose:
            print('Finished Training')

    def adversarial_train(self, bnn_net, attack, checkpoints=None, save_path_base=None):
        """
        Adversarially train the BNN using an attack.

        :param bnn_net: (nn.Module) The torch BNN model.
        :param attack: The desired adversarial attack class that implements a '.attack(net)' method.
        :param checkpoints: (list) List of epochs at which the model shall be saved.
        :param save_path_base: (str) The path to which the model shall be saved if a list of checkpoints is given.
        :return: None
        """
        if checkpoints is None:
            checkpoints = []
        cross_entropy = nn.CrossEntropyLoss()
        for epoch in range(self.n_epochs):
            running_loss = []
            for i, data in enumerate(self.train_loader, 0):
                original_inputs, labels = data[0].to(self.device), data[1].to(self.device)
                adversarial_inputs = attack.attack(bnn_net, original_inputs, labels).detach()
                self.optimizer.zero_grad()
                outputs = bnn_net(adversarial_inputs)
                loss = cross_entropy(outputs, labels) + bnn_net.kl_loss()
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
                torch.save(bnn_net, save_path_base + f'_epoch{epoch + 1}.pt')
        if self.verbose:
            print('Finished Training')


class FullyConnectedBNNTester:
    """
    Wrapper class to test BNNs.
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

    def test(self, bnn_net, n=10):
        """
        Calculates the test accuracy of the BNN on the test set. The accuracy is defined as the ratio between the number
        of correctly predicted labels and the total number of labels.

        :param bnn_net: (nn.Module) The torch BNN model.
        :param n: (int) Number of times we resample the weights and do a forward pass to obtain the mean output of the
            network.
        :return: (float) Test accuracy as the ratio between the correctly predicted labels and all labels.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = bnn_net.predict_class_probabilities(images, n)
                _, predicted = torch.max(outputs.data, 1)
                # for output, std, pred, label in zip(outputs, stds, predicted, labels):
                #     if pred != label:
                #         print(f'Predicted: {pred} Label: {label}')
                #         print(output, std)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if self.verbose:
            print(f'Accuracy of the network on the {total} test images: %d %%' % (100 * correct / total))
        return correct/total

    def adversarial_test(self, bnn_net, attack, n=10):
        """
        Calculates the robust accuracy for a BNN on a given attack.

        :param bnn_net: (nn.Module) The torch BNN model.
        :param attack: The desired adversarial attack class that implements a '.attack(net)' method.
        :param n: (int) Number of times we resample the weights and do a forward pass to obtain the mean output of the
            network.
        :return: (float) Robust accuracy as the average of 1-(relative frequency of successful attacks) for each class.
        """
        correct = 0
        total = 0
        for data in self.test_loader:
            original_inputs, labels = data[0].to(self.device), data[1].to(self.device)
            adversarial_inputs = attack.attack(bnn_net, original_inputs, labels)
            outputs = bnn_net.predict_class_probabilities(adversarial_inputs.detach(), n)
            _, predicted = torch.max(outputs.data, 1)
            # for output, std, pred, label in zip(outputs, stds, predicted, labels):
            #     if pred != label:
            #         print(f'Predicted: {pred} Label: {label}')
            #         print(output, std)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if self.verbose:
            print(f'Accuracy of the network on the {total} test images: %d %%' % (100 * correct / total))
        return correct/total
