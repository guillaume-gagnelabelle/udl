import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax, max_pool2d
import torch.nn.init as init
from utils import device

class VI(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, trainloader, nb_epochs=120, task_id=0):
        self.train()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        for epoch in range(nb_epochs):
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                self.optimizer.zero_grad()
                N = self.mc_samples
                targets = targets - 2 * task_id                    

                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.repeat(N, 1, 1, 1)    # shape = [N * B, channel, H, W]
                targets = targets.repeat(N).view(-1)
                
                outputs = self(inputs, task_id)

                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            if (epoch + 1) % 10 == 0: print('(%d/%d): Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch + 1, nb_epochs, train_loss/(batch_idx+1), 100.*correct/total, correct / N, total / N))

    def test(self, testloader, task_id):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(testloader):
                B = inputs.shape[0] # batch size
                N = 50              # MC samples
                targets = targets - 2 * task_id    # normalize the targets to be labelled 0 or 1

                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.repeat(N, 1, 1, 1)    # shape = [N * B, channel, H, W]
                
                outputs = self(inputs, task_id)
                outputs = softmax(outputs.view(N, B, -1), dim=2)
                outputs = torch.mean(outputs, 0)
                predicted = torch.argmax(outputs, dim=1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total
            return correct, total, acc
        
    def stack_param(self, layer):
        return torch.cat([p.view(-1) for p in layer.parameters()])

    def update_prior(self, task_id, new_task=False, initial=False):
        mu1 = self.stack_param(self.mu1)
        mu2 = self.stack_param(self.mu2)
        mu_head = []
        # first, accumulate the weights of the heads of the previous heads
        for i in range(task_id):
            mu_head.append(self.stack_param(self.mu_head[i]))
        # then, if it's a new task, we instantiate the prior to a normal (0, 1)
        if new_task:
            first_prior = torch.zeros_like(self.stack_param(self.mu_head[0]), device=device)
            mu_head.append(first_prior)
        # if it's not a new task, e.g. when training on coreset, we already have a prior
        else:
            head = self.stack_param(self.mu_head[task_id])
            mu_head.append(head)
        mu_head = torch.cat(mu_head)
        mu = torch.cat([mu1, mu2, mu_head])

        log_var1 = self.stack_param(self.log_var1)
        log_var2 = self.stack_param(self.log_var2)
        log_var_head = []
        for i in range(task_id):
            log_var_head.append(self.stack_param(self.log_var_head[i]))
        # if it's a new task (initial), we instantiate the prior to a normal (0, 1)
        if new_task:
            first_prior = torch.zeros_like(self.stack_param(self.log_var_head[0]), device=device)
            log_var_head.append(first_prior)
        # if it's not a new task, e.g. when training on coreset, we already have a prior
        else:
            head = self.stack_param(self.log_var_head[task_id])
            log_var_head.append(head)
        log_var_head = torch.cat(log_var_head)
        log_var = torch.cat([log_var1, log_var2, log_var_head])

        # if it's the initial prior, the whole model has a prior N(0,1), not just the new head
        if initial:
            self.prior_mu = torch.zeros_like(mu.detach())
            self.prior_log_var = torch.zeros_like(log_var.detach())
        else:
            self.prior_mu, self.prior_log_var = mu.detach(), log_var.detach()

        if new_task and not initial: self.task_id = task_id

    def compute_q(self, task_id):
        mu1 = self.stack_param(self.mu1)
        mu2 = self.stack_param(self.mu2)
        mu_head = []
        for i in range(self.task_id + 1):
            mu_head.append(self.stack_param(self.mu_head[i]))
        mu_head = torch.cat(mu_head)
        mu = torch.cat([mu1, mu2, mu_head])

        log_var1 = self.stack_param(self.log_var1)
        log_var2 = self.stack_param(self.log_var2)
        log_var_head = []
        for i in range(self.task_id + 1):
            log_var_head.append(self.stack_param(self.log_var_head[i]))
        log_var_head = torch.cat(log_var_head)
        log_var = torch.cat([log_var1, log_var2, log_var_head])
        
        return mu, log_var
    
    def kl_divergence(self):
        posterior_mu, posterior_log_var = self.compute_q(self.task_id)
        prior_mu, prior_log_var = self.prior_mu, self.prior_log_var
        prior_var = torch.exp(prior_log_var)

        a = prior_log_var - posterior_log_var
        b = (torch.exp(posterior_log_var) + (prior_mu - posterior_mu) ** 2) / prior_var - 1
        
        kl = 0.5 * torch.sum(a + b)
        return kl
    
    def set_task_size(self, task_size):
        self.task_size = task_size


    def initialize_weights_biases(self, model, nb_heads, sigma_0=-6, identical=False):
        if identical:
            # Initialize mean
            self.mu1.weight = nn.Parameter((model.mu1.weight).detach())
            self.mu1.bias = nn.Parameter((model.mu1.bias).detach())
            self.mu2.weight = nn.Parameter((model.mu2.weight).detach())
            self.mu2.bias = nn.Parameter((model.mu2.bias).detach())
            for i in range(nb_heads):
                self.mu_head[i].weight = nn.Parameter((model.mu_head[i].weight).detach())
                self.mu_head[i].bias = nn.Parameter((model.mu_head[i].bias).detach())
            self.log_var1.weight = nn.Parameter((model.log_var1.weight).detach())
            self.log_var1.bias = nn.Parameter((model.log_var1.bias).detach())
            self.log_var2.weight = nn.Parameter((model.log_var2.weight).detach())
            self.log_var2.bias = nn.Parameter((model.log_var2.bias).detach())
            for i in range(nb_heads):
                self.log_var_head[i].weight = nn.Parameter((model.log_var_head[i].weight).detach())
                self.log_var_head[i].bias = nn.Parameter((model.log_var_head[i].bias).detach())
        else:

            # Initialize mean
            self.mu1.weight = nn.Parameter((model.mu1.weight).detach())
            self.mu1.bias = nn.Parameter((model.mu1.bias).detach())
            self.mu2.weight = nn.Parameter((model.mu2.weight).detach())
            self.mu2.bias = nn.Parameter((model.mu2.bias).detach())
            for i in range(nb_heads):
                nn.init.normal_(self.mu_head[i].weight, mean=0, std=0.1)
                nn.init.normal_(self.mu_head[i].bias, mean=0, std=0.1)
            # initial variance = 1e-6 means log(1e-6) = -6
            init.constant_(self.log_var1.weight, sigma_0)
            init.constant_(self.log_var1.bias, sigma_0)
            init.constant_(self.log_var2.weight, sigma_0)
            init.constant_(self.log_var2.bias, sigma_0)
            for i in range(nb_heads):
                init.constant_(self.log_var_head[i].weight, sigma_0)
                init.constant_(self.log_var_head[i].bias, sigma_0)


class MLP_VI(VI):
    def __init__(self, in_dim, hidden_dim, out_dim, model_init, nb_heads, mc_samples=100, identical=False):
        super().__init__()

        self.mu1 = nn.Linear(in_dim, hidden_dim)
        self.mu2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = torch.nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(nb_heads)])

        self.log_var1 = Exponentiated_Noised_Linear(in_dim, hidden_dim)
        self.log_var2 = Exponentiated_Noised_Linear(hidden_dim, hidden_dim)
        self.log_var_head = torch.nn.ModuleList([Exponentiated_Noised_Linear(hidden_dim, out_dim) for _ in range(nb_heads)])

        if identical:
            self.initialize_weights_biases(model_init, sigma_0=-6, nb_heads=nb_heads, identical=True)
            self.prior_mu = model_init.prior_mu
            self.prior_log_var = model_init.prior_log_var
            self.task_id = model_init.task_id
            self.mc_samples = model_init.mc_samples

        else:
            self.initialize_weights_biases(model_init, sigma_0=-6, nb_heads=nb_heads)
            self.mc_samples = mc_samples
            self.task_id = 0

        self.to(device)

    def forward(self, x, task_id):
        x = x.view(x.shape[0], -1)
        
        x = self.mu1(x) + self.log_var1(x)
        x = relu(x)
        x = self.mu2(x) + self.log_var2(x)
        x = relu(x)

        x = self.mu_head[task_id](x) + self.log_var_head[task_id](x)

        return x
      
    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets) + self.kl_divergence() / self.task_size
    
    
class Exponentiated_Noised_Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        
    def forward(self, x):
        
        eps_w, eps_b = torch.randn_like(self.weight), torch.randn_like(self.bias)
            
        W = torch.exp(0.5*self.weight) * eps_w
        b = torch.exp(0.5*self.bias) * eps_b

        return x @ W.T + b

# Simple MLP to find the maximum likelihood of the first task
class MLP(VI):
    def __init__(self, in_dim, hidden_dim, out_dim, mc_samples=1):
        super().__init__()

        self.mu1 = nn.Linear(in_dim, hidden_dim)
        self.mu2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = torch.nn.ModuleList([nn.Linear(hidden_dim, out_dim)])

        self.mc_samples = mc_samples
        self.to(device)

    def forward(self, x, task_id=-1):
        x = x.view(x.shape[0], -1)

        x = self.mu1(x)
        x = relu(x)
        x = self.mu2(x)
        x = relu(x)
        x = self.mu_head[0](x)

        return x

    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)
    
################################### EXTENSION ###################################
class ConvNet_VI(VI):
    def __init__(self, in_channels, hidden_channels, out_channels, model_init, nb_heads, mc_samples=100, identical=False):
        super().__init__()
           
        self.mu1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.mu2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.mu_head = torch.nn.ModuleList([nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=0) for _ in range(nb_heads)])

        self.log_var1 = Exponentiated_Noised_Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.log_var2 = Exponentiated_Noised_Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.log_var_head = torch.nn.ModuleList([Exponentiated_Noised_Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=0) for _ in range(nb_heads)])

        if identical:
            self.initialize_weights_biases(model_init, sigma_0=-6, nb_heads=nb_heads, identical=True)
            self.prior_mu = model_init.prior_mu
            self.prior_log_var = model_init.prior_log_var
            self.task_id = model_init.task_id
            self.mc_samples = model_init.mc_samples

        else:
            self.initialize_weights_biases(model_init, sigma_0=-6, nb_heads=nb_heads)
            self.mc_samples = mc_samples
            self.task_id = 0
        
        self.to(device)

    def forward(self, x, task_id):

        x = self.mu1(x) + self.log_var1(x)
        x = relu(x)
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = self.mu2(x) + self.log_var2(x)
        x = relu(x)
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = self.mu_head[task_id](x) + self.log_var_head[task_id](x)
        x = x.squeeze()

        return x

    def loss_fn(self, outputs, targets):
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
        self.mu_head = torch.nn.ModuleList([nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=0)])

        self.mc_samples = 1
        self.to(device)

    def forward(self, x, task_id=-1):
        x = self.mu1(x)
        x = relu(x)
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = self.mu2(x)
        x = relu(x)
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = self.mu_head[0](x)
        x = x.squeeze()

        return x

    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)

################################################################################