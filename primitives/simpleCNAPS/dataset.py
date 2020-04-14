from PIL import Image
import torch
from torch.utils import data

__all__ = ('Dataset')

class Dataset(data.Dataset):
  # NumPy Dataset
  def __init__(self, datalist):
      self.datalist = datalist

  def __getitem__(self, index):
        """
        Generates one sample of data
        """
        sample_data = 0.0

        return sample_data

  def __len__(self):
        # Total Number of samples
        return len(self.all_data)
