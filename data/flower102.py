from pathlib import Path
from PIL import Image
from typing import Optional, Callable
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms as T

class Flower102(Dataset):
    def __init__(
        self,
        data_path: str = '../../Datasets/flower102/', # should have jpg and and lists inside. In lists, there should be train/val/test split CSVs
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Optional[int] = 256,
        all_in_ram: bool = False
    ):
        super(Flower102, self).__init__()
        assert split in ['train', 'val', 'test'], f"split should be train/val/test but given {split}"
        
        if split.lower() == 'train':
            csv_file = Path(data_path, 'lists/trn.csv')
        elif split.lower() == 'val':
            csv_file = Path(data_path, 'lists/val.csv')
        elif split.lower() == 'test':
            csv_file = Path(data_path, 'lists/tst.csv')
            
        data_df = pd.read_csv(csv_file)
        images = data_df['file_name'].tolist()
        labels = data_df['label'].tolist()
        cls_name_to_idx = np.sort(np.unique(labels))
        cls_name_to_idx = {k:v for k,v in enumerate(cls_name_to_idx)}
        cls_idx_to_name = {v:k for k,v in cls_name_to_idx.items()}
        
        self.dataset = []
        for idx in range(len(images)):
            img_path = Path(data_path, 'jpg/', images[idx])
            if all_in_ram:
                img_path = Image.open(img_path).convert("RGB")
                
            label = cls_name_to_idx[labels[idx]]
            self.dataset.append((img_path, label))
        
        
        if transform is None:
            transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        
        self.targets = [v for (_, v) in self.dataset]
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
    dataset = Flower102(split='test', all_in_ram=True)
    print(dataset[2][0].shape)
    print(dataset.get_class(5))
    # image, label = dataset[2]
    # print(image.shape)
    # print(label, dataset.cls_idx_to_name[label])