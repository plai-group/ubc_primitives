from PIL import Image
import torch
from torch.utils import data

__all__ = ('Dataset_1', 'Dataset_2')

class Dataset_1(data.Dataset):
  # NumPy Dataset
  def __init__(self, all_data_X, all_data_Y, use_labels):
      self.all_data_X = all_data_X
      self.use_labels = use_labels
      if self.use_labels:
          self.all_data_Y = all_data_Y

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


class Dataset_2(data.Dataset):
  # Image read Dataset
  def __init__(self, all_data, preprocess, use_labels):
      self.all_data    = all_data
      self.pre_process = preprocess
      self.use_labels  = use_labels
      if self.use_labels:
          self.all_data_Y = all_data_Y

  def __getitem__(self, index):
        """
        Generates one sample of data
        """
        # Select sample
        img_path, sample_label = self.all_data[index]

        # Load data and get label
        image = Image.open(img_path)
        image = self.pre_process(image)

        if self.use_labels:
            sample_label = float(sample_label)

            return image, sample_label

        return image

  def __len__(self):
        # Total Number of samples
        return len(self.all_data)
