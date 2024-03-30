import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax, max_pool2d
import torch.nn.init as init
from utils import device

class VI(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, trainloader, nb_epochs=100):
        self.train()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        for epoch in range(nb_epochs):
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                self.optimizer.zero_grad()
                N = self.mc_samples

                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.repeat(N, 1, 1, 1)    # shape = [N * B, channel, H, W]
                targets = targets.repeat(N).view(-1)
                
                outputs = self(inputs)

                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            if (epoch + 1) % 10 == 0: print('(%d/%d): Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch + 1, nb_epochs, train_loss/(batch_idx+1), 100.*correct/total, correct / N, total / N))

    def test(self, testloader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(testloader):
                B = inputs.shape[0] # batch size
                N = 50              # MC samples

                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.repeat(N, 1, 1, 1)    # shape = [N * B, channel, H, W]
                
                outputs = self(inputs)
                outputs = softmax(outputs.view(N, B, -1), dim=2)
                outputs = torch.mean(outputs, 0)
                predicted = torch.argmax(outputs, dim=1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total
            return correct, total, acc
        
    def stack_param(self, layer):
        return torch.cat([p.view(-1) for p in layer.parameters()])

    def update_prior(self, initial=False):
        mu, log_var = self.compute_q()

        # if it's the initial prior, the model has a prior N(0,1)
        if initial:
            self.prior_mu = torch.zeros_like(mu.detach())
            self.prior_log_var = torch.zeros_like(log_var.detach())
        else:
            self.prior_mu, self.prior_log_var = mu.detach(), log_var.detach()

    
    def compute_q(self):
        mu1 = self.stack_param(self.mu1)
        mu2 = self.stack_param(self.mu2)
        mu_head = self.stack_param(self.mu_head)
        mu = torch.cat([mu1, mu2, mu_head])

        log_var1 = self.stack_param(self.log_var1)
        log_var2 = self.stack_param(self.log_var2)
        log_var_head = self.stack_param(self.log_var_head)
        log_var = torch.cat([log_var1, log_var2, log_var_head])
        
        return mu, log_var
    
    def kl_divergence(self):
        posterior_mu, posterior_log_var = self.compute_q()
        prior_mu, prior_log_var = self.prior_mu, self.prior_log_var

        a = prior_log_var - posterior_log_var
        b = (torch.exp(posterior_log_var) + (prior_mu - posterior_mu) ** 2) / torch.exp(prior_log_var) - 1
        
        kl = 0.5 * torch.sum(a + b)
        return kl
    
    def set_task_size(self, task_size):
        self.task_size = task_size


    def initialize_weights_biases(self, model, sigma_0=-6, identical=False):
        if identical:
            self.mu1.weight = nn.Parameter((model.mu1.weight).detach())
            self.mu1.bias = nn.Parameter((model.mu1.bias).detach())
            self.mu2.weight = nn.Parameter((model.mu2.weight).detach())
            self.mu2.bias = nn.Parameter((model.mu2.bias).detach())
            self.mu_head.weight = nn.Parameter((model.mu_head.weight).detach())
            self.mu_head.bias = nn.Parameter((model.mu_head.bias).detach())

            self.log_var1.weight = nn.Parameter((model.log_var1.weight).detach())
            self.log_var1.bias = nn.Parameter((model.log_var1.bias).detach())
            self.log_var2.weight = nn.Parameter((model.log_var2.weight).detach())
            self.log_var2.bias = nn.Parameter((model.log_var2.bias).detach())
            self.log_var_head.weight = nn.Parameter((model.log_var_head.weight).detach())
            self.log_var_head.bias = nn.Parameter((model.log_var_head.bias).detach())

        else:

            # Initialize mean
            self.mu1.weight = nn.Parameter((model.mu1.weight).detach())
            self.mu1.bias = nn.Parameter((model.mu1.bias).detach())
            self.mu2.weight = nn.Parameter((model.mu2.weight).detach())
            self.mu2.bias = nn.Parameter((model.mu2.bias).detach())
            self.mu_head.weight = nn.Parameter((model.mu_head.weight).detach())
            self.mu_head.bias = nn.Parameter((model.mu_head.bias).detach())

            # initial variance = 1e-6 means log(1e-6) = -6
            init.constant_(self.log_var1.weight, sigma_0)
            init.constant_(self.log_var1.bias, sigma_0)
            init.constant_(self.log_var2.weight, sigma_0)
            init.constant_(self.log_var2.bias, sigma_0)
            init.constant_(self.log_var_head.weight, sigma_0)
            init.constant_(self.log_var_head.bias, sigma_0)


class MLP_VI(VI):
    def __init__(self, in_dim, hidden_dim, out_dim, model_init, mc_samples=100, identical=False):
        super().__init__()

        self.mu1 = nn.Linear(in_dim, hidden_dim)
        self.mu2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, out_dim)

        self.log_var1 = Exponentiated_Noised_Linear(in_dim, hidden_dim)
        self.log_var2 = Exponentiated_Noised_Linear(hidden_dim, hidden_dim)
        self.log_var_head = Exponentiated_Noised_Linear(hidden_dim, out_dim)

        if identical:
            self.initialize_weights_biases(model_init, identical=True)
            self.prior_mu = model_init.prior_mu
            self.prior_log_var = model_init.prior_log_var
            self.mc_samples = model_init.mc_samples

        else:
            self.initialize_weights_biases(model_init, sigma_0=-6)
            self.mc_samples = mc_samples
        
        self.to(device)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = self.mu1(x) + self.log_var1(x)
        x = relu(x)
        x = self.mu2(x) + self.log_var2(x)
        x = relu(x)
        x = self.mu_head(x) + self.log_var_head(x)

        return x
      
    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets) + self.kl_divergence() / self.task_size


class Exponentiated_Noised_Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        
    def forward(self, x):
        
        eps_w, eps_b = torch.randn_like(self.weight), torch.randn_like(self.bias)
            
        W = torch.exp(0.5 * self.weight) * eps_w
        b = torch.exp(0.5 * self.bias) * eps_b

        return x @ W.T + b

# Simple MLP to find the maximum likelihood of the first task
class MLP(VI):
    def __init__(self, in_dim, hidden_dim, out_dim, mc_samples=1):
        super().__init__()

        self.mu1 = nn.Linear(in_dim, hidden_dim)
        self.mu2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, out_dim)

        self.mc_samples = mc_samples
        self.to(device)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.mu1(x)
        x = relu(x)
        x = self.mu2(x)
        x = relu(x)
        x = self.mu_head(x)

        return x

    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)
    

################################### EXTENSION ###################################
class ConvNet_VI(VI):
    def __init__(self, in_channels, hidden_channels, out_channels, model_init, mc_samples=100, identical=False):
        super().__init__()
           
        self.mu1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.mu2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.mu_head = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=0)

        self.log_var1 = Exponentiated_Noised_Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.log_var2 = Exponentiated_Noised_Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.log_var_head = Exponentiated_Noised_Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=0)

        if identical:
            self.initialize_weights_biases(model_init, identical=True)
            self.prior_mu = model_init.prior_mu
            self.prior_log_var = model_init.prior_log_var
            self.mc_samples = model_init.mc_samples

        else:
            self.initialize_weights_biases(model_init, sigma_0=-6)
            self.mc_samples = mc_samples
        
        self.to(device)

    def forward(self, x):

        x = self.mu1(x) + self.log_var1(x)
        x = relu(x)
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = self.mu2(x) + self.log_var2(x)
        x = relu(x)
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = self.mu_head(x) + self.log_var_head(x)
        x = x.squeeze()

        return x

    def loss_fn(self, outputs, targets, print_loss=False):
        if print_loss:
            print("    xent= %.4f" % (nn.CrossEntropyLoss()(outputs, targets).item()))
            print("    kl  = %.4f" % ((self.kl_divergence() / self.task_size).item()))
            print("    loss= %.4f" % (nn.CrossEntropyLoss()(outputs, targets).item() +  (self.kl_divergence() / self.task_size).item()))
        return nn.CrossEntropyLoss()(outputs, targets) + self.kl_divergence() / self.task_size

class Exponentiated_Noised_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        eps_w, eps_b = torch.randn_like(self.weight), torch.randn_like(self.bias)

        W = torch.exp(0.5 * self.weight) * eps_w
        b = torch.exp(0.5 * self.bias) * eps_b

        x = nn.functional.conv2d(input=x, weight=W, bias=b, stride=self.stride, padding=self.padding)

        return x
    
# Simple ConvNet to find the maximum likelihood of the first task
class ConvNet(VI):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.mu1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.mu2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.mu_head = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=0)

        self.mc_samples = 1
        self.to(device)

    def forward(self, x):
        x = self.mu1(x)
        x = relu(x)
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = self.mu2(x)
        x = relu(x)
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = self.mu_head(x)
        x = x.squeeze()

        return x

    def loss_fn(self, outputs, targets, print_loss=False):
        if print_loss:
            print('xent = %.4f' % (nn.CrossEntropyLoss()(outputs, targets).item()))
        return nn.CrossEntropyLoss()(outputs, targets)

################################################################################