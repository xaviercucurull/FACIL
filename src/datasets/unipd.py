import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Senz3D(Dataset):
    """ UniPD Senz3D dataset.

    Args:
        split (str, optional): Dataset split to load. {"train", "val", "test"}. Defaults to "train".

        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Defaults to False.

        transform (callable, optional): A function/transform that  takes in an image
            and returns a transformed version.
    """

    # TODO: select number of classes / classes -> label encoder?

    # TODO: decide if train/val split could be done afterwards
    train_subjects = ['S1', 'S2']
    val_subjects = ['S3']
    test_subjects = ['S4']

    class_map = {"G1" : 0, "G2": 1, 'G3': 2, "G4" : 3, "G5": 4, 'G6': 5,
                 "G7" : 6, "G8": 7, 'G9': 8, "G10" : 9, "G11": 10}

    classes = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11"]
    
    landmark_dict_filename = 'unipd_handgest_senz3d.npy' 

    def __init__(self, split='train', data_path=None, download=False, transform=None):
        if split == "val":
            subjects = self.val_subjects
        elif split == "test":
            subjects = self.test_subjects
        else:
            subjects = self.train_subjects

        if download:
            self.download()
        
        # Load landmarks dict
        if data_path:
                self.landmark_dict = np.load(os.path.join(data_path, self.landmark_dict_filename), allow_pickle=True)
        else:
            self.landmark_dict = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'UniPd', self.landmark_dict_filename), allow_pickle=True)

        # Dataset name
        dataset_name = list(self.landmark_dict.keys())[0]

        # From dict to data array (x, y)
        self.data = []
        self.targets = []

        for s in subjects:
            for g, i in self.landmark_dict[dataset_name][s].items():
                for filename, landmark_list in i.items():
                    self.data.append(landmark_list)
                    self.targets.append(self.class_map[g])

        self._calc_distfeatures()

    def __len__(self):
        return len(self.data)

    def _calc_distfeatures(self):
        self.features = []

        for landmark_list in self.data:
            # Calculate feature vector (Distances)
            landmark_num = len(landmark_list)//2
            landmark_dist_features = []
            for i in range(landmark_num):
                for j in range(landmark_num):
                    if j>i:
                        landmark_dist_features.append(landmark_list[2*i] - landmark_list[2*j])      # x
                        landmark_dist_features.append(landmark_list[2*i+1] - landmark_list[2*j+1])  # y

            self.features.append(landmark_dist_features)

        self.features = np.array(self.features, dtype=np.float32)

    def download(self):
        """ Download dataset from the internet and extract landmarks.
        """
        # TODO
        return