from PIL import Image
import torch
from torch.utils import data

__all__ = ('Dataset_1', 'Dataset_2')

class Dataset_1(data.Dataset):
  # Dataset
  def __init__(self, all_data_X, all_data_Y, use_labels):
      self.all_data_X = all_data_X
      self.all_data_Y = all_data_Y
      self.use_labels = use_labels
      # Check if class index is from 0 to C-1 or 1 to C
      if (0.0 in all_data_Y[:, 0]) or (0 in all_data_Y[:, 0]):
          sub_class_index = False
      else:
          sub_class_index = True

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
  # Dataset
  def __init__(self, all_data, preprocess, use_labels):
      self.all_data    = all_data
      self.pre_process = preprocess
      # Check if class index is from 0 to C-1 or 1 to C
      if (0.0 in all_data[:, 0]) or (0 in all_data_Y[:, 0]):
          sub_class_index = False
      else:
          sub_class_index = True

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
