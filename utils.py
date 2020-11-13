import torch
import numpy as np


class UMAPDataset:

    def __init__(self, data, epochs_per_sample, head, tail, weight, device='cpu', batch_size=1000):

        """
        create dataset for iteration on graph edges

        """
        self.weigh = weight
        self.batch_size = batch_size
        self.data = data
        self.device = device

        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        self.num_edges = len(self.edges_to_exp)

        # shuffle edges
        shuffle_mask = np.random.permutation(range(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask]
        self.edges_from_exp = self.edges_from_exp[shuffle_mask]

    def get_batches(self):
        batches_per_epoch = int(self.num_edges / self.batch_size / 5)
        for _ in range(batches_per_epoch):
            rand_index = np.random.randint(0, len(self.edges_to_exp) - 1, size=self.batch_size)
            batch_index_to = self.edges_to_exp[rand_index]
            batch_index_from = self.edges_from_exp[rand_index]
            if self.device == 'cuda':
                batch_to = torch.Tensor(self.data[batch_index_to]).cuda()
                batch_from = torch.Tensor(self.data[batch_index_from]).cuda()
            else:
                batch_to = torch.Tensor(self.data[batch_index_to])
                batch_from = torch.Tensor(self.data[batch_index_from])
            yield (batch_to, batch_from)