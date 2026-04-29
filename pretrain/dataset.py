import io
import h5py
from typing import Callable, Optional
from PIL import Image
import json
import os


class PathologyDataset:
    def __init__(self, root: str = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, h5_root: str = None) -> None:
        print(f"Loading dataset on {h5_root} according to {root} ...")
        self.h5_root = h5_root
        self.images = json.load(open(root, "r", encoding="utf-8"))
        print(f"Loaded {len(self.images)} images from {root}")
        self.transformers = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        try:
            image_idx, h5 = self.images[index]
            h5 = os.path.join(self.h5_root, h5)
            with h5py.File(h5, "r") as hf:
                try:
                    img_bytes = hf['patches'][image_idx].tobytes()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except:
                    img_bytes = hf[image_idx][:].tobytes()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            img = Image.new("RGB", (224, 224), (255, 255, 255))
            # Safe logging for multiprocessing
            try:
                with open("./failed_path.txt", "a") as f:
                    f.write(f"Error loading image {image_idx} from {h5}: {e}\n")
            except:
                print(f"Error loading image {image_idx} from {h5}: {e}")
        
        if self.transformers is not None:
            img = self.transformers(img)

            
        return img, 0
