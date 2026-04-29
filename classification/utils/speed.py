import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from multiprocessing.pool import Pool

def load_pt_file(path):
    '''Load a PyTorch tensor from a .pt file.'''
    try:
        return torch.load(path, weights_only=True), os.path.basename(path)
    except Exception as e:
        print(f"[dataset] Cannot load features from {path}: {e}")
        return None, None

def load_multi_slides(roots, datasets, slides, feature):
    '''
    Load features for multiple slides from specified datasets in multiprocessing manner.
    if slides contains over 5 slides, we sample 1024 patches from each slide and return indices in order to load coords later.
    '''
    features = []
    dict_slide2coords = {}
    success_slides = []
    try:
        if len(slides.split("/")) > 5:
            # if slides contains over 5 slides, we sample 1024 patches from each slide
            use_sampling = True
        else:
            use_sampling = False

        # Prepare arguments for multiprocessing
        args_list = []
        for slide in slides.split("/"):
            for dataset in datasets.split("/"):
                root = roots[dataset]
                feature_path = os.path.join(root, "pt_files", feature, os.path.splitext(slide)[0] + ".pt")
                if os.path.exists(feature_path):
                    break
            args_list.append(feature_path)
        # Load features in parallel
        with Pool(8) as pool:
            results = pool.map(load_pt_file, args_list)
        
        success_slides = [slide for _, slide in results if slide is not None]
        for feature, slide_name in results:
            if feature is not None:
                if use_sampling and feature.shape[0] > 1024:
                    indices = np.random.choice(feature.shape[0], 1024, replace=False)
                    feature = feature[indices]
                    dict_slide2coords[slide_name.split('.pt')[0]] = indices
                features.append(feature)
            else:
                print(f"[dataset] Feature for slide {slide_name} could not be loaded.")

        if len(features) == 0:
            raise ValueError("[dataset] No features found for slides: {}".format(slides))
        features = torch.cat(features, dim=0)
    except Exception as e:
        print(f"[dataset] Error loading features for slides {slides}: {e}")
    return features, success_slides, dict_slide2coords

def load_features(roots, datasets, slides, feature):
    
    features = []
    try:
        for slide in slides.split("/"):
            for dataset in datasets.split("/"):
                root = roots[dataset]
                if os.path.exists(os.path.join(root, "pt_files", feature, os.path.splitext(slide)[0] + ".pt")):
                    break
            try:
                features.append(torch.load(os.path.join(root, "pt_files", feature, os.path.splitext(slide)[0] + ".pt"), weights_only=True))
            except Exception as e:
                print(f"[dataset] Cannot load features for slide {slide}")
        if len(features) == 0:
            raise ValueError("No features for slide %s" % slides)
        features = torch.cat(features, dim=0)
    except Exception as e:
        print(e)
        raise ValueError("Cannot load features for slide %s" % slides)
    return features


def load_coords(roots, datasets, slides, feature):
    coords = []
    try:
        for slide in slides.split("/"):
            for dataset in datasets.split("/"):
                root = roots[dataset]
                if os.path.exists(os.path.join(root, "patches", os.path.splitext(slide)[0] + ".h5")):
                    break
            try:
                coord = h5py.File(os.path.join(root, "patches", os.path.splitext(slide)[0] + ".h5"), "r")["coords"]
                coord = np.array(coord)
                coords.append(torch.tensor(coord))
            except Exception as e:
                print(f"[dataset] Cannot load coords for slide {slide}")
        if len(coords) == 0:
            raise ValueError("No coords for slide %s" % slides)
        coords = torch.cat(coords, dim=0)
    except Exception as e:
        print(f"[dataset] Cannot load coords for slide {slides}")
        # raise ValueError("Cannot load coords for slide %s" % slides)
        coords = torch.zeros((0, 2))  # Return empty tensor if coords cannot be loaded
    return coords


def load_files(row, roots, feature):
    dataset_, case_, slide_ = row["dataset"], row["case"], row["slide"]
    features = load_features(roots, dataset_, slide_, feature)  # load pt file
    # features, success_slides, indices_coords = load_multi_slides(roots, dataset_, slide_, feature)  # load pt file
    coords = load_coords(roots, dataset_, slide_, features)  # load h5 file
    # coords = load_coords(roots, dataset_, success_slides, indices_coords)  # load h5 file
    return (case_, slide_, features, coords)


def load_files_wrapper(args):
    return load_files(*args)


def load_into_memory(datasets, rows, feature, num_processes):
    # df = pd.read_excel(datasets)
    df = pd.read_excel(datasets, sheet_name="feature status", header=1, skiprows=range(0, 1))
    roots = dict()
    for idx, row in df.iterrows():
        dataset = row["Dataset"]
        if dataset not in roots.keys() and not pd.isna(dataset):
            roots[dataset] = row["Feature Path"]

    cases, slides, features, coords = [None] * len(rows), [None] * len(rows), [None] * len(rows), [None] * len(rows)
    print("[dataset] loading dataset into memory")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(load_files_wrapper, [(row, roots, feature) for idx, row in rows.iterrows()]), total=len(rows), desc="Loading files"))

    for idx, result in enumerate(results):
        cases[idx], slides[idx], features[idx], coords[idx] = result

    print("[dataset] loading done")
    for idx in range(len(cases)):
        if cases[idx] != rows["case"].values.tolist()[idx]:
            raise ValueError("The sequence of cases is not the same as the original excel file")
    return cases, slides, features, coords
