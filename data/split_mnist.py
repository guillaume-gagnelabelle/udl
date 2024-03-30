from torchvision import datasets
import torch

class Split_MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, coreset_size=40):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.full_data = self.data
        self.full_targets = self.targets

        self.coreset_size = coreset_size

    # select_task does not update the coreset size. Allows us to loop over the dataset.
    def select_task(self, task_id, update_coreset=False):

        if update_coreset: 
            self.update_random_coreset(task_id)
        mask = (self.full_targets == 2 * task_id) | (self.full_targets == 2 * task_id + 1)
        self.data = self.full_data[mask]
        self.targets = self.full_targets[mask]

        return self.targets.shape[0]    # task size


    # This function has to be called before updating to the next task to ensure that the images&labels
    # are removed from full_data before computing data
    def update_random_coreset(self, task_id):
        if self.coreset_size > 0:
            labels_to_select = torch.tensor([2 * task_id, 2 * task_id + 1])

            indices_to_select = torch.where(torch.isin(self.full_targets, labels_to_select))[0]
            shuffled_indices = torch.randperm(len(indices_to_select))
            selected_indices = indices_to_select[shuffled_indices[:self.coreset_size]]

            # Update coreset_data and coreset_targets for the task_id
            if task_id == 0:
                self.coreset_data = self.full_data[selected_indices]
                self.coreset_targets = self.full_targets[selected_indices]
            else:
                self.coreset_data = torch.cat([self.coreset_data, self.full_data[selected_indices]])
                self.coreset_targets = torch.cat([self.coreset_targets, self.full_targets[selected_indices]])

            # Remove selected samples from full_data and full_targets
            self.full_data = torch.cat([self.full_data[idx].unsqueeze(0) for idx in range(len(self.full_data)) if idx not in selected_indices], dim=0)
            self.full_targets = torch.cat([self.full_targets[idx].unsqueeze(0) for idx in range(len(self.full_targets)) if idx not in selected_indices], dim=0)
            
    # To be called at the end of training to finetune the model
    def train_on_coreset(self):
        self.buffer_data = self.full_data
        self.buffer_targets = self.full_targets
        self.full_data = self.coreset_data
        self.full_targets = self.coreset_targets

    def train_on_full_dataset(self):
        self.full_data = self.buffer_data
        self.full_targets = self.buffer_targets
