import torch
from torch.utils.data import Dataset, random_split
import random
from itertools import product

class RHM_Dataset(Dataset):
    """
    Random Hierarchical Model Dataset.
    Generates a synthetic tree-based grammar dataset.
    """
    def __init__(self, num_samples, L=3, s=2, m=7, v=14, num_classes=2, seed=42):
        self.num_samples = num_samples
        self.L = L
        self.s = s
        self.m = m
        self.v = v
        self.num_classes = num_classes

        self.rules = self._generate_grammar(seed)
        self.data, self.labels = self._generate_dataset(seed)

    def _generate_grammar(self, seed):
        """Generates the hierarchical grammar rules."""
        random.seed(seed)
        # Cartesian product for unique tuples
        tuples = list(product(*[range(self.v) for _ in range(self.s)]))
        rules = {}
        # Level 0 (root)
        rules[0] = torch.tensor(random.sample(tuples, self.num_classes * self.m)).reshape(self.num_classes, self.m, -1)
        # The other layers
        for i in range(1, self.L):
            rules[i] = torch.tensor(random.sample(tuples, self.v * self.m)).reshape(self.v, self.m, -1)
        return rules

    def _generate_dataset(self, seed):
        """Generates sequences based on the grammar rules."""
        torch.manual_seed(seed)
        trees = {}
        # Creation of random roots (Classes)
        labels = torch.randint(low=0, high=self.num_classes, size=(self.num_samples,))
        trees[0] = labels.clone()

        for l in range(self.L):
            # Random choice 1 from m synonyms
            chosen_rule = torch.randint(low=0, high=self.m, size=labels.shape)
            labels = self.rules[l][labels, chosen_rule].flatten(start_dim=1)
            trees[l+1] = labels.clone()

        return trees[self.L].long(), trees[0].long()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloaders(num_samples=20000, L=3, s=2, m=7, v=14, num_classes=2, seed=42, batch_size=64, train_size=10000, test_size=10000):
    """Utility function to get train and test dataloaders."""
    full_dataset = RHM_Dataset(num_samples=num_samples, L=L, s=s, m=m, v=v, num_classes=num_classes, seed=seed)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, full_dataset
