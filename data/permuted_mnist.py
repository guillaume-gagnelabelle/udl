from torchvision import datasets
import torch

class Permuted_MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, coreset_size=200):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.full_data = self.data
        self.full_targets = self.targets

        self.coreset_size = coreset_size
        self.train_coreset_bool = False
        self.set_permutations()

    def set_permutations(self):
        torch.manual_seed(0)    # this way the trainset and the testset contain the same permutation tensor
        nb_pixels = 28*28
        self.permutations = torch.stack([torch.randperm(nb_pixels) for _ in range(10)])
        

    def select_task(self, task_id, update_coreset=False):
        if update_coreset: 
            self.update_random_coreset(task_id)

        if not self.train_coreset_bool:
            permuted_data = self.full_data.view(self.full_data.shape[0], -1)[:, self.permutations[task_id]]

            self.data = permuted_data.view(self.full_data.shape)
            self.targets = self.full_targets
        else:
            self.data = self.coreset_data
            self.targets = self.coreset_targets

        return self.targets.shape[0]    # task size
    

    # This function has to be called before updating to the next task to ensure that the images&labels
    # are removed from full_data before computing data
    def update_random_coreset(self, task_id):
        if self.coreset_size > 0:

            random_indices = torch.randperm(self.data.shape[0])
            selected_indices = random_indices[:self.coreset_size]

            # Update coreset_data and coreset_targets for the task_id
            if task_id == 0:
                self.coreset_data = self.data[selected_indices]
                self.coreset_targets = self.targets[selected_indices]
            else:
                self.coreset_data = torch.cat([self.coreset_data, self.data[selected_indices]])
                self.coreset_targets = torch.cat([self.coreset_targets, self.targets[selected_indices]])

    # To be called at the end of training to finetune the model with the coreset
    def train_on_coreset(self):
        self.train_coreset_bool = True

    def train_on_full_dataset(self):
        self.train_coreset_bool = False
