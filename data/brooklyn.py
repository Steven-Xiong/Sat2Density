import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from abc import ABC, abstractmethod
import h5py
import io
# import sparse
import imageio.v2 as imageio
import torchvision.transforms.functional as TF
import scipy

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

def imread(path):
  image = imageio.imread(path)
  return image
def preprocess(image):

  image = image / 255.0
  #image = np.where(image >= 0, image, 0)
  return image

# class BrooklynQueensDataset(Dataset):
#   """Brooklyn and Queens Dataset."""

#   def __init__(self, name, mode="train", neighbors=20):
#     self.mode = mode

#     if any(x in name for x in ['brooklyn', 'queens']):
#       name, label = name.split('_')
#       local_name = name.split('-')[0]
#       data_dir = "/home/x.zhexiao/near-remote/src/data/brooklyn_queens/"
#       self.aerial_dir = "{}{}/overhead/images/".format(data_dir, local_name)  #"{}{}/aerial/".format(data_dir, local_name)
#       self.label_dir = "{}{}/labels/{}/".format(data_dir, local_name, label)
#       self.streetview_dir = "{}{}/streetview/".format(data_dir, local_name)
#     else:
#       raise ValueError('Unknown dataset.')

#     self.name = name
#     self.label = label
#     self.base_dir = "/home/x.zhexiao/near-remote/src/data/brooklyn_queens/"
#     self.config = self.setup(name, label, neighbors)

#     self.h5_name = "{}_train.h5".format(name) if mode in [
#         "train", "val"
#     ] else "{}_test.h5".format(name)

#     tmp_h5 = h5py.File("{}{}/{}".format(self.base_dir, self.name, self.h5_name),
#                        'r')
#     self.dataset_len = len(tmp_h5['fname'])

#     if self.mode != "test":
#       # use part of training for validation
#       np.random.seed(1)
#       inds = np.random.permutation(list(range(0, self.dataset_len)))

#       K = 500
#       if self.mode == "train":
#         self.dataset_len = self.dataset_len - K
#         self.inds = inds[:self.dataset_len]
#       elif self.mode == "val":
#         self.inds = inds[self.dataset_len - K:]
#         self.dataset_len = K

#   def setup(self, name, label, neighbors):
#     config = {}
#     config['loss'] = "cross_entropy"

#     # adjust output size
#     #import pdb; pdb.set_trace()
#     if label == 'age':
#       config['num_output'] = 15
#       config['ignore_label'] = [0, 1]
#     elif label == 'function':
#       config['num_output'] = 208
#       config['ignore_label'] = [0, 1]
#     elif label == 'landuse':
#       config['num_output'] = 13
#       config['ignore_label'] = [1]
#     elif label == 'landcover':
#       config['num_output'] = 9
#       config['ignore_label'] = [0]
#     elif label == 'height':
#       config['num_output'] = 2
#       config['loss'] = "uncertainty"
#     else:
#       raise ValueError('Unknown label.')

#     # setup neighbors
#     config['near_size'] = neighbors

#     return config

#   def open_hdf5(self):
#     self.h5_file = h5py.File(
#         "{}{}/{}".format(self.base_dir, self.name, self.h5_name), "r")

#   def open_streetview(self):
#     fname = "panos_256*1024_new.h5"  #panos_calibrated_small.h5"
#     self.sv_file = h5py.File("{}{}".format(self.streetview_dir, fname), "r")

#   def __getitem__(self, idx):
#     if not hasattr(self, 'h5_file'):
#       self.open_hdf5()
#     if not hasattr(self, 'sv_file'):
#       self.open_streetview()

#     if self.mode != "test":
#       idx = self.inds[idx]

#     fname = self.h5_file['fname'][idx]
#     bbox = self.h5_file['bbox'][idx]
#     label = self.h5_file['label'][idx]
#     near_inds = self.h5_file['near_inds'][idx].astype(int)

#     # from matlab to python indexing
#     near_inds = near_inds - 1

#     # setup neighbors
#     if 0 < self.config['near_size'] <= near_inds.shape[-1]:
#       near_inds = near_inds[:self.config['near_size']]
#     else:
#       raise ValueError('Invalid neighbor size.')

#     # near locs, near feats
#     sort_index = np.argsort(near_inds)
#     unsort_index = np.argsort(sort_index)
#     near_locs = self.h5_file['locs'][near_inds[sort_index], ...][unsort_index,
#                                                                  ...]

#     # decode and preprocess panoramas
#     near_streetview = self.sv_file['images'][near_inds[sort_index],
#                                              ...][unsort_index, ...]
#     tmp = []
#     for item in near_streetview:
#       tmp_im = utils.preprocess(imageio.imread(io.BytesIO(item))).transpose(
#           2, 0, 1)
#       tmp_im_t = torch.from_numpy(tmp_im).float()
#       tmp_im_t_norm = TF.normalize(tmp_im_t,
#                                    mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#       tmp.append(tmp_im_t_norm)
#     near_streetview = torch.stack(tmp, dim=0)

#     # form absolute paths
#     fname_image = "{}{}".format(self.aerial_dir, fname.decode())
#     fname_label = "{}{}".format(self.label_dir, label.decode())

#     image = utils.preprocess(utils.imread(fname_image))
#     if self.label == "height":
#       fname_label = "{}.npz".format(fname_label[:-4])
#       label = sparse.load_npz(fname_label).todense()
#       label = label * (1200 / 3937)  # 1 ft (US survey) = 1200/3937 m
#       label = label - label.min()
#     else:
#       label = utils.imread(fname_label)

#     t_image = TF.to_tensor(image).float()
#     t_label = torch.from_numpy(label).float()
#     t_bbox = torch.from_numpy(bbox).float()
#     t_near_locs = torch.from_numpy(near_locs).float()
#     t_near_images = near_streetview

#     return {'A':t_image, 'B': t_near_images[0]}

#   def __len__(self):
#     return self.dataset_len

class BaseDataset(Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass
def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        # dataset_class.getitem(self,1)
        self.dataset = dataset_class(opt)

        print("dataset [%s] was created" % type(self.dataset).__name__)
        # import pdb; pdb.set_trace()
        # print('workers',int(opt.num_threads))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=16)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

# not aligned
# class Dataset():

#     def __init__(self,mode):
#     #   BaseDataset.__init__(self, opt)
#       self.dirA = os.path.join('/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens','brooklyn/overhead/images/19') # phase 另算
#       self.dirB = os.path.join('/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens','brooklyn/streetview/images')
#       self.dirD = os.path.join('/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens','brooklyn/streetview/seg_fast')
#       #self.dir_seg = os.path.join()
#       name = 'brooklyn-fc8_landuse'
#       neighbors=20
#       #import pdb; pdb.set_trace()
#       if any(x in name for x in ['brooklyn', 'queens']):
      
#         name, label = name.split('_')    # brooklyn-fc8  landcover
#         local_name = name.split('-')[0]
#         #import pdb; pdb.set_trace()
#         data_dir = '/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens/' #"/u/eag-d1/data/near-remote/"
#         self.aerial_dir = "{}{}/overhead/images/".format(data_dir, local_name)  #"{}{}/aerial/".format(data_dir, local_name)
#         self.label_dir = "{}{}/labels/{}/".format(data_dir, local_name, label)
#         self.streetview_dir = "{}{}/streetview/".format(data_dir, local_name)
#         self.streetview_seg_dir = "{}{}/streetview/seg_fast/".format(data_dir, local_name)
#       else:
#         raise ValueError('Unknown dataset.')
      
#       self.name = name
#       self.label = label
#       self.base_dir = "/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens/"  #"/u/eag-d1/scratch/scott/learn/near-remote/data/"
#       self.config = self.setup(name, label, neighbors)
      
#       #import pdb; pdb.set_trace()
#       self.h5_name = "{}_train.h5".format(name) if mode in [
#           "train", "val"
#       ] else "{}_test.h5".format(name)

#       tmp_h5 = h5py.File("{}{}/{}".format(self.base_dir, self.name, self.h5_name),
#                         'r')
#       self.dataset_len = len(tmp_h5['fname'])
#       self.mode = mode
#       if self.mode != "test":
#       # use part of training for validation
#         np.random.seed(1)
#         inds = np.random.permutation(list(range(0, self.dataset_len)))

#         K = 500
#         #self.mode = 'train'
#         if self.mode == "train":
#           self.dataset_len = self.dataset_len - K
#           self.inds = inds[:self.dataset_len]
#         elif self.mode == "val":
#           self.inds = inds[self.dataset_len - K:]
#           self.dataset_len = K

#     def setup(self, name, label, neighbors):
#       config = {}
#       config['loss'] = "cross_entropy"
#       #import pdb; pdb.set_trace()
#       # adjust output size
#       if label == 'age':
#         config['num_output'] = 15
#         config['ignore_label'] = [0, 1]
#       elif label == 'function':
#         config['num_output'] = 208
#         config['ignore_label'] = [0, 1]
#       elif label == 'landuse':
#         config['num_output'] = 13
#         config['ignore_label'] = [1]
#       elif label == 'landcover':
#         config['num_output'] = 9
#         config['ignore_label'] = [0]
#       elif label == 'height':
#         config['num_output'] = 2
#         config['loss'] = "uncertainty"
#       else:
#         raise ValueError('Unknown label.')

#       # setup neighbors
#       config['near_size'] = neighbors

#       return config

#     def open_hdf5(self):
#       self.h5_file = h5py.File(
#         "{}{}/{}".format(self.base_dir, self.name, self.h5_name), "r")

#     def open_streetview(self):
#       fname = 'panos_256*1024_new.h5'   #"panos_calibrated_small.h5"
#       fname_seg = 'seg_256*1024.h5'
#       #import pdb; pdb.set_trace()
#       self.sv_file = h5py.File("{}{}".format(self.streetview_dir, fname), "r")
#       self.sv_file_seg = h5py.File("{}{}".format(self.streetview_dir, fname_seg), "r")
#     def open_streetview_seg(self):
#       fname = 'seg_256*1024.h5'
#       self.sv_file_seg = h5py.File("{}{}".format(self.streetview_dir, fname), "r")

#     def __getitem__(self,idx):
#       """Return a data point and its metadata information.

#         Parameters:
#             index - - a random integer for data indexing

#         Returns a dictionary that contains A, B, A_paths and B_paths
#             A (tensor) - - an image in the input domain
#             B (tensor) - - its corresponding image in the target domain
#             A_paths (str) - - image paths
#             B_paths (str) - - image paths (same as A_paths)
#         """
      
#       # import pdb; pdb.set_trace()
#       if not hasattr(self, 'h5_file'):
#         self.open_hdf5()
        
#       if not hasattr(self, 'sv_file'):
#         self.open_streetview()
#         #self.open_streetview_seg()
        
#       if self.mode != "test":
#         idx = self.inds[idx]
#       # import pdb; pdb.set_trace()
#       fname = self.h5_file['fname'][idx]     
#       bbox = self.h5_file['bbox'][idx]
#       label = self.h5_file['label'][idx]
#       near_inds = self.h5_file['near_inds'][idx].astype(int)

#       # from matlab to python indexing
#       near_inds = near_inds - 1

#       # setup neighbors
#       if 0 < self.config['near_size'] <= near_inds.shape[-1]:  # 20 closest street-level panoramas
#         near_inds = near_inds[:self.config['near_size']]
#       else:
#         raise ValueError('Invalid neighbor size.')

#       # near locs, near feats
#       sort_index = np.argsort(near_inds)            #搜索对应的最近的index
#       unsort_index = np.argsort(sort_index)
#       near_locs = self.h5_file['locs'][near_inds[sort_index], ...][unsort_index,
#                                                                   ...]

#       # decode and preprocess panoramas
#       near_streetview = self.sv_file['images'][near_inds[sort_index],
#                                               ...][unsort_index, ...]
#       #near_streetview1 = near_streetview.astype(float)
#       near_streetview_seg = self.sv_file_seg['images'][near_inds[sort_index],
#                                               ...][unsort_index, ...]

#       tmp = []
#       for item in near_streetview:
#         tmp_im = preprocess(imageio.imread(io.BytesIO(item))).transpose(
#             2, 0, 1)
#         tmp_im_t = torch.from_numpy(tmp_im).float()
#         tmp_im_t_norm = TF.normalize(tmp_im_t,
#                                     mean=[0.5, 0.5, 0.5],
#                                     std=[0.5, 0.5, 0.5])
#         tmp.append(tmp_im_t)
#       near_streetview = torch.stack(tmp, dim=0)
      
#       tmp_seg = []
#       for item in near_streetview_seg:
#         tmp_im_seg = preprocess(imageio.imread(io.BytesIO(item))).transpose(
#             2, 0, 1)
#         tmp_im_t_seg = torch.from_numpy(tmp_im_seg).float()
#         tmp_im_t_norm_seg = TF.normalize(tmp_im_t_seg,
#                                     mean=[0.5, 0.5, 0.5],
#                                     std=[0.5, 0.5, 0.5])
#         tmp_seg.append(tmp_im_t_seg)
#       near_streetview_seg = torch.stack(tmp_seg, dim=0)

#       # form absolute paths
#       fname_image = "{}{}".format(self.aerial_dir, fname.decode())
#       fname_label = "{}{}".format(self.label_dir, label.decode())
#       fname_pano_seg = "{}{}".format(self.streetview_seg_dir, fname.decode().replace('19/',''))
#       image = preprocess(imread(fname_image))  # array
      
#       if self.label == "height":
#         fname_label = "{}.npz".format(fname_label[:-4])
#         label = scipy.sparse.load_npz(fname_label).todense()
#         label = label * (1200 / 3937)  # 1 ft (US survey) = 1200/3937 m
#         label = label - label.min()
#       else:
#         label = imread(fname_label)
  

#       t_image = TF.to_tensor(image).float()
#       t_label = torch.from_numpy(label).float()
#       t_bbox = torch.from_numpy(bbox).float()
#       t_near_locs = torch.from_numpy(near_locs).float()
#       t_near_images = near_streetview
#       #import pdb; pdb.set_trace()
      
#       t_pano_seg = near_streetview_seg
#       # paste all 4 satellite images together, transfer shape [256,256] to [256,1024]
      
#       for i in range(1, 4):
#             #t_image = torch.cat((t_image, transforms.ToTensor()(image).float()),2)
#             #image= np.ascontiguousarray(image)
#             #rot_image= image.transpose(1,0,2)[::-1]
            
#             rot_image = image #np.rot90(image,i)
#             rot_image = np.ascontiguousarray(rot_image)
#             t_image = torch.cat((t_image, transforms.ToTensor()(rot_image).float()),2)

#       source_image = t_near_images[1]
      
#       target_image = t_near_images[0]
#       target_loc = t_near_locs[0]
#       source_loc = t_near_locs[1]
#       #source_loc1 = t_near_locs[2]
#       '''
#       disp_vec = np.asarray([target_loc[0]-source_loc[0], target_loc[1]-source_loc[1]])
#       vec = make_gaussian_vector(disp_vec[0], disp_vec[1])
      
#       ###################
#       #if cfg.data_align:
#       theta_x = (180.0 / np.pi) * np.arctan2(disp_vec[1], disp_vec[0])  # first y and then x i.e. arctand (y/x)

#       # angle from y-axis or north
#       theta_y = 90 + theta_x

#       if theta_y < 0:  # fixing negative
#           theta_y += 360

#       column_shift = np.int(
#           theta_y * (cfg.data.image_size[1]/360.0) )   

#       source_image = torch.roll(source_image, column_shift, dims=2)  # rotate columns
#       target_image = torch.roll(target_image, column_shift, dims=2)  # rotate columns
      
#       #################
#       # source_image = AddBorder_tensor(source_image, cfg.data.border_size) # border_size = 0 may led to fault
#       # target_image = AddBorder_tensor(target_image, cfg.data.border_size)
#       '''
#       source_image1 = source_image
#       input = {}
      
#       input['sat'] = TF.to_tensor(image).float()#.permute(1,2,0)
#       input['pano'] = t_near_images[0]#.permute(1,2,0)
#       input['paths'] = []

#       # input_a  = (pano*sky+black_ground*(1-sky))
#       # for idx in range(len(input_a)):
#       #     if idx == 0:
#       #         sky_histc = input_a[idx].histc()[10:]
#       #     else:
#       #         sky_histc = torch.cat([input_a[idx].histc()[10:],sky_histc],dim=0)
#       # input['sky_histc'] = sky_histc
#       return input # A是target, B是condition
  
#     def __len__(self):
#       return self.dataset_len
  
  


# aligned
class Dataset():

    def __init__(self,mode):
    #   BaseDataset.__init__(self, opt)
      self.dirA = os.path.join('/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens','brooklyn/overhead') # phase 另算
      self.dirB = os.path.join('/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens','brooklyn/streetview/images')
      # self.dirD = os.path.join('/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens','brooklyn/streetview/seg_fast')
      #self.dir_seg = os.path.join()
      name = 'brooklyn-fc8_landuse'
      neighbors=20
      #import pdb; pdb.set_trace()
      if any(x in name for x in ['brooklyn', 'queens']):
      
        name, label = name.split('_')    # brooklyn-fc8  landcover
        local_name = name.split('-')[0]
        #import pdb; pdb.set_trace()
        data_dir = '/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens/' #"/u/eag-d1/data/near-remote/"
        self.aerial_dir = "{}{}/overhead/".format(data_dir, local_name)  #"{}{}/aerial/".format(data_dir, local_name)
        self.label_dir = "{}{}/labels/{}/".format(data_dir, local_name, label)
        self.streetview_dir = "{}{}/streetview/".format(data_dir, local_name)
        self.streetview_seg_dir = "{}{}/streetview/seg_fast/".format(data_dir, local_name)
      else:
        raise ValueError('Unknown dataset.')
      
      self.name = name
      self.label = label
      self.base_dir = "/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens/"  #"/u/eag-d1/scratch/scott/learn/near-remote/data/"
      self.config = self.setup(name, label, neighbors)
      
      #import pdb; pdb.set_trace()
      self.h5_name = "{}_train.h5".format(name) if mode in [
          "train", "val"
      ] else "{}_test.h5".format(name)

      tmp_h5 = h5py.File("{}{}/{}".format(self.base_dir, self.name, self.h5_name),
                        'r')
      self.dataset_len = len(tmp_h5['fname'])
      self.mode = mode
      if self.mode != "test":
      # use part of training for validation
        np.random.seed(1)
        inds = np.random.permutation(list(range(0, self.dataset_len)))

        K = 500
        #self.mode = 'train'
        if self.mode == "train":
          self.dataset_len = self.dataset_len - K
          self.inds = inds[:self.dataset_len]
        elif self.mode == "val":
          self.inds = inds[self.dataset_len - K:]
          self.dataset_len = K

    def setup(self, name, label, neighbors):
      config = {}
      config['loss'] = "cross_entropy"
      #import pdb; pdb.set_trace()
      # adjust output size
      if label == 'age':
        config['num_output'] = 15
        config['ignore_label'] = [0, 1]
      elif label == 'function':
        config['num_output'] = 208
        config['ignore_label'] = [0, 1]
      elif label == 'landuse':
        config['num_output'] = 13
        config['ignore_label'] = [1]
      elif label == 'landcover':
        config['num_output'] = 9
        config['ignore_label'] = [0]
      elif label == 'height':
        config['num_output'] = 2
        config['loss'] = "uncertainty"
      else:
        raise ValueError('Unknown label.')

      # setup neighbors
      config['near_size'] = neighbors

      return config

    def open_hdf5(self):
      self.h5_file = h5py.File(
        "{}{}/{}".format(self.base_dir, self.name, self.h5_name), "r")

    def open_streetview(self):
      fname = 'panos_256*1024_new.h5'   #"panos_calibrated_small.h5"
      fname_seg = 'seg_256*1024.h5'
      fname_sat_aligned = 'sat_aligned.h5'
      #import pdb; pdb.set_trace()
      self.sv_file = h5py.File("{}{}".format(self.streetview_dir, fname), "r")
      self.sv_file_seg = h5py.File("{}{}".format(self.streetview_dir, fname_seg), "r")
      self.sv_file_sat_aligned = h5py.File("{}{}".format(self.aerial_dir, fname_sat_aligned), "r")
    def open_streetview_seg(self):
      fname = 'seg_256*1024.h5'
      self.sv_file_seg = h5py.File("{}{}".format(self.streetview_dir, fname), "r")

    def __getitem__(self,idx):
      """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
      
      # import pdb; pdb.set_trace()
      if not hasattr(self, 'h5_file'):
        self.open_hdf5()
        
      if not hasattr(self, 'sv_file'):
        self.open_streetview()
        #self.open_streetview_seg()
        
      if self.mode != "test":
        idx = self.inds[idx]
      # import pdb; pdb.set_trace()
      fname = self.h5_file['fname'][idx]     
      bbox = self.h5_file['bbox'][idx]
      label = self.h5_file['label'][idx]
      near_inds = self.h5_file['near_inds'][idx].astype(int)

      # from matlab to python indexing
      near_inds = near_inds - 1

      # setup neighbors
      if 0 < self.config['near_size'] <= near_inds.shape[-1]:  # 20 closest street-level panoramas
        near_inds = near_inds[:self.config['near_size']]
      else:
        raise ValueError('Invalid neighbor size.')

      # near locs, near feats
      sort_index = np.argsort(near_inds)            #搜索对应的最近的index
      unsort_index = np.argsort(sort_index)
      near_locs = self.h5_file['locs'][near_inds[sort_index], ...][unsort_index,
                                                                  ...]

      # decode and preprocess panoramas
      near_streetview = self.sv_file['images'][near_inds[sort_index],
                                              ...][unsort_index, ...]
      #near_streetview1 = near_streetview.astype(float)
      near_streetview_seg = self.sv_file_seg['images'][near_inds[sort_index],
                                              ...][unsort_index, ...]
      # add aligned satellite
      sat_aligned = self.sv_file_sat_aligned['images'][near_inds[sort_index],
                                              ...][unsort_index, ...]

      tmp = []
      for item in near_streetview:
        tmp_im = preprocess(imageio.imread(io.BytesIO(item))).transpose(
            2, 0, 1)
        tmp_im_t = torch.from_numpy(tmp_im).float()                  #11.15:不用normalization了
        tmp_im_t_norm = TF.normalize(tmp_im_t,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        tmp.append(tmp_im_t)
      near_streetview = torch.stack(tmp, dim=0)
      
      tmp_seg = []
      for item in near_streetview_seg:
        tmp_im_seg = preprocess(imageio.imread(io.BytesIO(item))).transpose(
            2, 0, 1)
        tmp_im_t_seg = torch.from_numpy(tmp_im_seg).float()
        tmp_im_t_norm_seg = TF.normalize(tmp_im_t_seg,               #11.15:不用normalization了
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        tmp_seg.append(tmp_im_t_seg)
      
      near_streetview_seg = torch.stack(tmp_seg, dim=0)
      tmp_sat_aligned = []
      for item in sat_aligned:
        tmp_im_sat_aligned = preprocess(imageio.imread(io.BytesIO(item))).transpose(
            2, 0, 1)
        tmp_im_t_sat_aligned = torch.from_numpy(tmp_im_sat_aligned).float()
        tmp_im_t_norm_sat_aligned = TF.normalize(tmp_im_t_sat_aligned,         #11.15:不用normalization了
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        tmp_sat_aligned.append(tmp_im_t_sat_aligned)
      sat_aligned = torch.stack(tmp_sat_aligned, dim=0)
      

      # form absolute paths
      fname_image = "{}{}".format(self.aerial_dir, fname.decode())
      fname_label = "{}{}".format(self.label_dir, label.decode())
      fname_pano_seg = "{}{}".format(self.streetview_seg_dir, fname.decode().replace('19/',''))
      # image = preprocess(imread(fname_image))  # array
      
      if self.label == "height":
        fname_label = "{}.npz".format(fname_label[:-4])
        label = scipy.sparse.load_npz(fname_label).todense()
        label = label * (1200 / 3937)  # 1 ft (US survey) = 1200/3937 m
        label = label - label.min()
      else:
        label = imread(fname_label)
  
      # import pdb; pdb.set_trace()
      t_image = sat_aligned[0]  #最临近的panorama
      t_label = torch.from_numpy(label).float()
      t_bbox = torch.from_numpy(bbox).float()
      t_near_locs = torch.from_numpy(near_locs).float()
      t_near_images = near_streetview
      #import pdb; pdb.set_trace()
      
      t_pano_seg = near_streetview_seg
      # paste all 4 satellite images together, transfer shape [256,256] to [256,1024]
      
      '''
      t_tensors = [t_image,t_image,t_image,t_image]
      t_image = torch.cat(t_tensors,2)
      '''
      # for i in range(1, 4):
      #       #t_image = torch.cat((t_image, transforms.ToTensor()(image).float()),2)
      #       #image= np.ascontiguousarray(image)
      #       #rot_image= image.transpose(1,0,2)[::-1]
            
      #       rot_image = t_image #np.rot90(image,i)
      #       rot_image = np.ascontiguousarray(rot_image).transpose(1,2,0)
      #       t_image = torch.cat((t_image, transforms.ToTensor()(rot_image).float()),2)

      source_image = t_near_images[1]
      
      target_image = t_near_images[0]
      target_loc = t_near_locs[0]
      source_loc = t_near_locs[1]
      #source_loc1 = t_near_locs[2]
      '''
      disp_vec = np.asarray([target_loc[0]-source_loc[0], target_loc[1]-source_loc[1]])
      vec = make_gaussian_vector(disp_vec[0], disp_vec[1])
      
      ###################
      #if cfg.data_align:
      theta_x = (180.0 / np.pi) * np.arctan2(disp_vec[1], disp_vec[0])  # first y and then x i.e. arctand (y/x)

      # angle from y-axis or north
      theta_y = 90 + theta_x

      if theta_y < 0:  # fixing negative
          theta_y += 360

      column_shift = np.int(
          theta_y * (cfg.data.image_size[1]/360.0) )   

      source_image = torch.roll(source_image, column_shift, dims=2)  # rotate columns
      target_image = torch.roll(target_image, column_shift, dims=2)  # rotate columns
      
      #################
      # source_image = AddBorder_tensor(source_image, cfg.data.border_size) # border_size = 0 may led to fault
      # target_image = AddBorder_tensor(target_image, cfg.data.border_size)
      '''
      # import pdb; pdb.set_trace()
      source_image1 = source_image
      input = {}
      
      input['sat'] = t_image #TF.to_tensor(t_image).float()#.permute(1,2,0)
      input['pano'] = t_near_images[0]#.permute(1,2,0)
      input['paths'] = []

      # input_a  = (pano*sky+black_ground*(1-sky))
      # for idx in range(len(input_a)):
      #     if idx == 0:
      #         sky_histc = input_a[idx].histc()[10:]
      #     else:
      #         sky_histc = torch.cat([input_a[idx].histc()[10:],sky_histc],dim=0)
      # input['sky_histc'] = sky_histc
      return input # A是target, B是condition
  
    def __len__(self):
      return self.dataset_len
  
  
