import os
import random
import torch

from torch.utils.data import Dataset

from utils import load_npz



class UnpairedDataset(Dataset):
    #get unpaired dataset, such as MR-CT dataset
    def __init__(self, A_path, B_path, is_random=False):
        # FIXME: Do something with randomness
        listA = os.listdir(A_path)
        listB = os.listdir(B_path)
        self.listA = [os.path.join(A_path,k) for k in listA]
        self.listB = [os.path.join(B_path,k) for k in listB]
        self.Asize = len(self.listA)
        self.Bsize = len(self.listB)
        self.dataset_size = max(self.Asize,self.Bsize)

    def __getitem__(self,index):
        if self.Asize == self.dataset_size:
            A,A_gt = load_npz(self.listA[index])
            B,B_gt = load_npz(self.listB[random.randint(0, self.Bsize - 1)])
        else :
            B,B_gt = load_npz(self.listB[index])
            A,A_gt = load_npz(self.listA[random.randint(0, self.Asize - 1)])


        A = torch.from_numpy(A.copy()).unsqueeze(0).float()
        A_gt = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()
        B = torch.from_numpy(B.copy()).unsqueeze(0).float()
        B_gt = torch.from_numpy(B_gt.copy()).unsqueeze(0).float()
        return A,A_gt,B,B_gt

    def __len__(self):
        return self.dataset_size


class SingleDataset(Dataset):
    def __init__(self,test_path):
        test_list = os.listdir(test_path)
        self.test = [os.path.join(test_path,k) for k in test_list]

    def __getitem__(self,index):
        img,gt = load_npz(self.test[index])

        img = torch.from_numpy(img.copy()).unsqueeze(0).float()
        gt = torch.from_numpy(gt.copy()).unsqueeze(0).float()
        return img, gt

    def __len__(self):
        return len(self.test)
