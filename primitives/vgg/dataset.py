from PIL import Image
import torch
from torch.utils import data

__all__ = ('Dataset',)

class Dataset(data.Dataset):
  # Dataset
  def __init__(self, all_data, preprocess):
      self.all_data    = all_data
      self.pre_process = preprocess

  def __getitem__(self, index):
        """
        Generates one sample of data
        """
        # Select sample
        img_path, label = self.all_data[index]

        # Load data and get label
        image = Image.open(img_path)
        image = self.pre_process(image)
        label = float(label)

        return image, label

  def __len__(self):
        # Total Number of samples
        return len(self.all_data)
