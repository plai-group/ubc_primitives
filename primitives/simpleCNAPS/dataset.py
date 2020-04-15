from PIL import Image
import torch
from torch.utils import data

__all__ = ('Dataset')

class Dataset(data.Dataset):
  def __init__(self, datalist, base_dir, mode="Train"):
      self.base_dir   = base_dir
      self.datalist   = datalist
      self._preproces = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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

  def __prepare_task(self, all_support_set, all_query_set):
      context_images = []
      context_labels = []
      target_images  = []
      target_labels  = []

      # Get support set
      for supp_set in all_support_set:
          supp_image_path = os.join.path(base_dir, supp_set[0])
          supp_image = Image.open(supp_image_path)
          supp_image = self.pre_process(supp_image)
          context_images.append(supp_image)
          context_labels.append(int(supp_set[1]))

      # Get target set
      for qury_set in all_query_set:
          qury_image_path = os.join.path(base_dir, qury_set[0])
          qury_image = Image.open(qury_image_path)
          qury_image = self.pre_process(qury_image)
          target_images.append(qury_image)
          target_labels.append(int(qury_set[1]))

    context_images = torch.FloatTensor(context_images)
    context_labels = torch.LongTensor(context_labels)
    target_images  = torch.FloatTensor(target_images)
    target_labels  = torch.LongTensor(target_labels)

    return context_images, target_images, context_labels, target_labels

  def __getitem__(self, index):
      """
      Generates one task sample of data
      """
      all_support_set, all_query_set = self.__get_task_set(index=(index+self.start_idx))
      context_images, target_images, context_labels, target_labels = self.__prepare_task(all_support_set, all_query_set)

      return context_images, target_images, context_labels, target_labels

  def __len__(self):
      # Total Number of tasks
      return len(self.all_set)
