import torch
from torch.utils import data

__all__ = ('Dataset',)

class Dataset(data.Dataset):
  # Dataset
  def __init__(self, all_data_X, all_data_Y, use_labels):
      self.all_data_X = all_data_X
      self.all_data_Y = all_data_Y
      self.use_labels = use_labels

  def __getitem__(self, index):
        """
        Generates one sample of data
        """
        # Select sample
        sample_data = self.all_data_X[index]
        # Load data and get label
        sample_data = torch.from_numpy(sample_data).float()

        if self.use_labels:
            sample_label = self.all_data_Y[index]
            sample_label = float(sample_label)

            return sample_data, sample_label

        return sample_data

  def __len__(self):
        # Total Number of samples
        return len(self.all_data_X)
