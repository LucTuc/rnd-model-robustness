import torch
import torch.nn as nn


class PGD:
    """
    A class implementing a traditional untargeted l-infinity PGD attack. Note that it can generate adversarial examples
    also in batches.
    """
    def __init__(self, n_iterations, epsilon, step_size, clamp=True, EOT=False, n=10):
        """
        The PGD object constructor. Ensure that the model is already on the target device.

        :param n_iterations: (int) The number of PGD steps to make for the attack.
        :param epsilon: (float) The radius of the l-infinity ball to which the attack is constrained.
        :param step_size: (float) The step size at each iteration.
        :param clamp: (bool) Set to 'True' if you wish to intersect the l-infinity ball of perturbations with the 748
            dimensional [0, 1] box (hardcoded for MNIST for now).
        :param EOT: (bool) Set to 'True' if you wish to perform an Expectation over Transformation (EOT) like attack.
            This is to counter obfuscated gradients caused by randomized models.
        :param n: (int) Number of Monte Carlo samples to estimate the expected gradient of a model. Only relevant in
            case we perform an EOT attack on a randomized model (self.EOT is 'True').
        """
        self.n_iterations = n_iterations
        self.epsilon = epsilon
        self.step_size = step_size
        self.clamp = clamp
        self.EOT = EOT
        self.n = n

    def _fgsm_untargeted(self, model, x, label):
        """
        A private method of the PGD class implementing an untargeted l-infinity FGSM attack. Ensure that 'x', 'label'
        and 'self.model' are on the same device. It is capable to process a batch of data.

        :param model: (nn.Module) The pytorch model that is to be attacked.
        :param x: (torch.Tensor) The input (batch) of data.
        :param label: (torch.Tensor) The label(s) of the input (batch).
        :return: (torch.Tensor) The produced (batch of) adversarial example(s) after one iteration of untargeted FGSM.
        """
        # the code is a bit long, but it is more memory efficient than just one loop
        # first run through
        input_ = x.clone().detach()
        input_.requires_grad_()
        logits = model.forward(input_)
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(logits, label)
        loss.backward()
        grad = input_.grad
        # if we take more than just one sample than we have to cumulate the gradient: EOT attack
        if self.EOT:
            for _ in range(self.n - 1):
                input_ = x.clone().detach()
                input_.requires_grad_()
                logits = model.forward(input_)
                model.zero_grad()
                loss = nn.CrossEntropyLoss()(logits, label)
                loss.backward()
                grad += input_.grad
        out = input_ + self.step_size * grad.sign()  # FGSM
        if self.clamp:
            out = out.clamp_(min=0, max=1)  # MNIST values hardcoded
        return out

    def _no_sign_attack_untargeted(self, model, x, label):
        # the code is a bit long, but it is more memory efficient than just one loop
        # first run through
        input_ = x.clone().detach()
        input_.requires_grad_()
        logits = model.forward(input_)
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(logits, label)
        loss.backward()
        grad = input_.grad
        # if we take more than just one sample than we have to cumulate the gradient: EOT attack
        if self.EOT:
            for _ in range(self.n - 1):
                input_ = x.clone().detach()
                input_.requires_grad_()
                logits = model.forward(input_)
                model.zero_grad()
                loss = nn.CrossEntropyLoss()(logits, label)
                loss.backward()
                grad += input_.grad
        out = input_ + self.step_size * grad * 1/self.n
        if self.clamp:
            out = out.clamp_(min=0, max=1)  # MNIST values hardcoded
        return out

    def attack(self, model, x, label):
        """
        The attack method of the PGD class. Implements an untargeted attack with the parameters given in the
        constructor. It can process the data both in batches and individually. Ensure that 'x', 'label'
        and 'self.model' are on the same device.

        :param model: (nn.Module) The pytorch model that is to be attacked.
        :param x: (torch.Tensor) The input (batch) of data.
        :param label: (torch.Tensor) The label(s) of the input (batch).
        :return: (torch.Tensor) The produced (batch of) adversarial example(s).
        """
        # the borders of the projection ball
        x_min = x - self.epsilon
        x_max = x + self.epsilon

        # choose a random starting point
        x = x + self.epsilon * (2 * torch.rand_like(x) - 1)
        x = x.clamp_(min=0, max=1)  # MNIST values hardcoded

        for i in range(self.n_iterations):
            # untargeted FGSM
            x = self._fgsm_untargeted(model, x, label)
            # no sign gradient step
            # x = self._no_sign_attack_untargeted(model, x, label)
            # project
            x = torch.min(torch.max(x_min, x), x_max)

        x = x.clamp_(min=0, max=1)

        return x.detach()
