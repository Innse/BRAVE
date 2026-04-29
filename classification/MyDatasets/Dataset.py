import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, all_datasets, feature, split_file):
        """
        Args:
            all_datasets (str): excel file with root information of each dataset
            feature (str): feature type, such as "resnet50" or "uni"
            split_file (str): excel file with split information
        """
        self.all_datasets = pd.read_excel(all_datasets, sheet_name="feature status", header=0)
        self.feature = feature
        self.split_file = pd.read_excel(split_file).reset_index(drop=True)
        print(self.split_file)
        datasets = self.all_datasets["Dataset"].unique()
        datasets = [dataset for dataset in datasets if not pd.isna(dataset)]
        self.roots = {dataset: self.all_datasets[self.all_datasets["Dataset"] == dataset]["Feature Path"].values[0] for dataset in datasets}
        for key, value in self.roots.items():
            if key not in self.split_file["dataset"].unique():
                continue
            if not os.path.exists(value):
                raise ValueError("Feature root %s does not exist" % value)
            else:
                print("[dataset] dataset <%s> from %s" % (key, value))
        # number of classes
        self.classes = self.split_file["label"].unique()
        print(f"[dataset] number of classes={len(self.classes)}: ({self.classes})")
        for label in self.classes:
            print(f"[label] {label}: {self.split_file[self.split_file['label'] == label]['class'].unique()}")
        # number of samples in each split
        self.splits = {split: [i for i, x in enumerate(self.split_file["split"].values.tolist()) if x == split] for split in self.split_file["split"].unique()}
        for key, value in self.splits.items():
            print(f"[dataset] number of cases in {key} split={len(value)}")
        # feature dimension
        try:
            filename = os.path.splitext(self.split_file["slide"].values[0].split("/")[0])[0] + ".pt"
            self.n_features = torch.load(
                os.path.join(self.roots[self.split_file["dataset"].values[0]], "pt_files", self.feature, filename),
                weights_only=True,
            ).shape[-1]
        except:
            raise ValueError("Feature dimension cannot be determined")
        print(f"[dataset] dimension of feature <{self.feature}>={self.n_features}")
        # pre-load the dataset into memory
        self.features, self.coords = None, None

    def __load_features__(self, datasets, slides):
        features = []
        use_sampling = False
        indices_coords = []
        success_loaded_slides = []
        try:
            if len(slides.split("/")) > 5:
                # if slides contains over 5 slides, we sample 1024 patches from each slide
                use_sampling = True
            for slide in slides.split("/"):
                # find root directory for this slide
                for dataset in datasets.split("/"):
                    root = self.roots[dataset]
                    if os.path.exists(os.path.join(root, "pt_files", self.feature, os.path.splitext(slide)[0] + ".pt")):
                        break
                try:
                    feature = torch.load(os.path.join(root, "pt_files", self.feature, os.path.splitext(slide)[0] + ".pt"), weights_only=True)
                    indices = np.arange(feature.shape[0])
                    # if use_sampling is True, we sample 1024 patches from the feature
                    if use_sampling and feature.shape[0] > 1024:
                        indices = np.random.choice(feature.shape[0], 1024, replace=False)
                        feature = feature[indices]
                        
                    features.append(feature)
                    indices_coords.append(indices)
                    success_loaded_slides.append(slide)
                except Exception as e:
                    print(f"[dataset] Error loading features for slide {slide}: {e}")
            if len(features) == 0:
                raise ValueError("[dataset] there is no feature for slide %s" % slides)
            features = torch.cat(features, dim=0)
        except Exception as e:
            print(f"[dataset] Error loading features for slide {slides}: {e}")
        return features, success_loaded_slides, indices_coords

    def __load_coords__(self, datasets, slides, indices_coords):
        coords = []
        try:
            for islide, slide in enumerate(slides):
                for dataset in datasets.split("/"):
                    root = self.roots[dataset]
                    if os.path.exists(os.path.join(root, "patches", os.path.splitext(slide)[0] + ".h5")):
                        break
                try:
                    coord = h5py.File(os.path.join(root, "patches", os.path.splitext(slide)[0] + ".h5"), "r")["coords"]
                    coord = np.array(coord)
                    coord = coord[indices_coords[islide]]
                    coords.append(torch.tensor(coord))
                except:
                    pass
                        # print(f"[dataset] Cannot load coords for slide {slide}")
            if len(coords) == 0:
                raise ValueError("Cannot load coords for slide %s" % slides)
            coords = torch.cat(coords, dim=0)
        except Exception as e:
            # print(f"[dataset] Error loading coords for slide {slides}: {e}")
            # raise ValueError("Cannot load coords for slide %s" % slides)
            coords = torch.zeros((0, 2))  # Return empty tensor if coords cannot be loaded
        return coords


    def __getitem__(self, idx):
        row = self.split_file.iloc[idx]
        [datasets, case, slide, classname, label, split] = row.values.tolist()
        if self.features is not None and self.coords is not None:
            features, coords = self.features[idx], self.coords[idx]
            if len(slide.split("/")) > 5:
                # if slides contains over 5 slides, we sample 1024 patches from each slide
                indices = np.random.choice(features.shape[0], 1024, replace=False)
                features = features[indices]
                coords = coords[indices]
        else:
            features, success_loaded_slides, indices_coords = self.__load_features__(datasets, slide)
            coords = self.__load_coords__(datasets, success_loaded_slides, indices_coords)
        return datasets, case, slide, classname, features, coords, label, split

    def __len__(self):
        return len(self.split_file)