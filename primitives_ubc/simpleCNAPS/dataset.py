import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms

__all__ = ('Dataset')

class Dataset(data.Dataset):
  def __init__(self, datalist, base_dir, mode="TRAIN"):
      self.mode       = mode
      self.base_dir   = base_dir
      self.datalist   = datalist
      self._preproces = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5),\
                                                                 (0.5, 0.5, 0.5))])
      self.__curate_datalist()

  def __curate_datalist(self):
      # Get set
      all_data = {}
      for idx in range(self.datalist.shape[0]):
          _, _set, img_path, label = list(self.datalist.iloc[idx, :])
          _set  = int(_set)
          label = int(label)
          if _set in all_data.keys():
              all_data[_set].append([img_path, label])
          else:
              all_data[_set] = [[img_path, label]]
      self.all_data = all_data
      self.all_set  = sorted(all_data.keys())
      # Check if set starts with 0 or 1
      if self.all_set[0] == 1:
          start_idx = 1
      else:
          start_idx = 0
      self.start_idx  = start_idx

  def __get_task_set(self, index):
      all_query_set   = []
      all_support_set = []

      sample_data = self.all_data[index]
      for data in sample_data:
          image_path, label = data
          if "support_set" in image_path:
              all_support_set.append([image_path, label])
          elif "query_set" in image_path:
              all_query_set.append([image_path, label])

      return all_support_set, all_query_set

  def __shuffle_set(self, images, labels):
      permutation = np.random.permutation(images.shape[0])

      return images[permutation], labels[permutation]

  def __prepare_task(self, all_support_set, all_query_set):
      context_images = []
      context_labels = []
      target_images  = []
      target_labels  = []

      # Get support set
      for supp_set in all_support_set:
          supp_image_path = os.path.join(self.base_dir, supp_set[0])
          supp_image = Image.open(supp_image_path)
          supp_image = self._preproces(supp_image)
          supp_image = torch.unsqueeze(supp_image, dim=0)
          supp_label = torch.tensor([int(supp_set[1])])
          supp_label = torch.unsqueeze(supp_label, dim=0)
          context_images.append(supp_image)
          context_labels.append(supp_label)

      # Get target set
      for qury_set in all_query_set:
          qury_image_path = os.path.join(self.base_dir, qury_set[0])
          qury_image = Image.open(qury_image_path)
          qury_image = self._preproces(qury_image)
          qury_image = torch.unsqueeze(qury_image, dim=0)
          qury_label = torch.tensor([int(qury_set[1])])
          qury_label = torch.unsqueeze(qury_label, dim=0)
          target_images.append(qury_image)
          target_labels.append(qury_label)

      context_images = (torch.cat(context_images, dim=0)).float()
      context_labels = (torch.cat(context_labels, dim=0)).long()
      target_images  = (torch.cat(target_images, dim=0)).float()
      target_labels  = (torch.cat(target_labels, dim=0)).long()

      return context_images, target_images, context_labels, target_labels

  def __getitem__(self, index):
      """
      Generates one task sample of data
      """
      all_support_set, all_query_set = self.__get_task_set(index=(index+self.start_idx))
      context_images, target_images, context_labels, target_labels = self.__prepare_task(all_support_set, all_query_set)

      if self.mode == "TRAIN":
          context_images, context_labels = self.__shuffle_set(images=context_images, labels=context_labels)
          target_images,  target_labels  = self.__shuffle_set(images=target_images,  labels=target_labels)

      return context_images, target_images, context_labels, target_labels

  def __len__(self):
      # Total Number of tasks
      return len(self.all_set)
