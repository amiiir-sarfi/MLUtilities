from pathlib import Path
from PIL import Image
from typing import Optional, Callable
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms as T

class MIT67(Dataset):
    def __init__(
        self,
        data_path: str = '../../Datasets/MIT/', # should have jpg and and lists inside. In lists, there should be train/val/test split CSVs
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Optional[int] = 256,
        all_in_ram: bool = False
    ):
        super(MIT67, self).__init__()
        assert split in ['train', 'val'], f"split should be train/val but given {split}"
        data_path = Path(data_path)

        anno = "TrainImages.txt" if split == 'train' else "TestImages.txt"
        paths = pd.read_csv(data_path/anno, header=None)
        
        paths.columns = ["paths"]
        
        filenames = paths["paths"].tolist()
        
        self.dataset = []
        targets = []
        for idx, img_name in enumerate(filenames):
            img_path = data_path/"Images"/img_name
            if all_in_ram:
                img_path = Image.open(img_path).convert("RGB")
            label = img_name.split('/')[0]
            targets.append(label)
            self.dataset.append((img_path, label))
        
        cls_name_to_idx = {clsname:idx for idx, clsname in enumerate(np.sort(np.unique(targets)))}
        cls_idx_to_name = {idx:clsname for clsname, idx in cls_name_to_idx.items()}
        
        if transform is None:
            transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        
        self.targets = [cls_name_to_idx[target] for target in targets]
        self.all_in_ram = all_in_ram
        self.cls_name_to_idx = cls_name_to_idx
        self.cls_idx_to_name = cls_idx_to_name
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if not self.all_in_ram:
            img = Image.open(img).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, self.cls_name_to_idx[label]
            
    def __len__(self):
        return len(self.dataset)   

if __name__ == "__main__":
    dataset = MIT67(split='train', all_in_ram=True)
    print(dataset[3][0].shape, dataset[3][1])
    # print(dataset.cls_name_to_idx)