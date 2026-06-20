from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List, Union, Optional

from ultr_ai.dataset.patient_level import PatientLevelDataset
from ultr_ai.dataset.utils import collate_patient_batch, SimpleVideoTransforms, TemporallyConsistentTransforms

class LungUltrasoundDataModule:
    
    def __init__(self, 
                 root_dir: str,
                 labels_csv: str,
                 file_metadata_csv: str,
                 image_folder: str = 'images',
                 video_folder: str = 'videos',
                 split_csv: Optional[str] = None,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 frame_sampling: int = 16,
                 depth_filter: str = 'all',
                 cache_size: int = 100,
                 files_per_site: Optional[Union[int, str]] = 'all',  
                 site_order: Optional[List[str]] = None,   
                 pad_missing_sites: bool = True,           
                 max_sites: Optional[int] = None):
        
        self.root_dir = root_dir
        self.labels_csv = labels_csv
        self.file_metadata_csv = file_metadata_csv
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.split_csv = split_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frame_sampling = frame_sampling
        self.cache_size = cache_size
        self.depth_filter = depth_filter
        self.files_per_site = files_per_site
        self.site_order = site_order
        self.pad_missing_sites = pad_missing_sites
        self.max_sites = max_sites
        
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.video_transforms = SimpleVideoTransforms()

        self.train_video_transforms = TemporallyConsistentTransforms(
            degrees=25,
            translate=(0.15, 0.15),
            scale=(0.65, 1.45),
            brightness=0.3,
            contrast=0.3,
            blur_sigma=(0.1, 0.5),
            noise_std=0.2,
            augment_prob=0.5,
            blur_prob=0.2,
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225]
        )
    
    def setup(self, stage=None):
        if stage == 'patient_level' or stage is None:
            self.patient_train = PatientLevelDataset(
                root_dir=self.root_dir,
                labels_csv=self.labels_csv,
                file_metadata_csv=self.file_metadata_csv,
                image_folder=self.image_folder,
                video_folder=self.video_folder,
                split='train',
                split_csv=self.split_csv,
                image_transforms=self.image_transforms,
                video_transforms=self.train_video_transforms,
                mode='video',
                frame_sampling=self.frame_sampling,
                depth_filter=self.depth_filter,
                cache_size=self.cache_size,
                files_per_site=self.files_per_site,
                site_order=self.site_order,
                pad_missing_sites=self.pad_missing_sites,
                max_sites=self.max_sites
            )
            
            self.patient_val = PatientLevelDataset(
                root_dir=self.root_dir,
                labels_csv=self.labels_csv,
                file_metadata_csv=self.file_metadata_csv,
                image_folder=self.image_folder,
                video_folder=self.video_folder,
                split='val',
                split_csv=self.split_csv,
                image_transforms=self.image_transforms,
                video_transforms=self.video_transforms,
                mode='video',
                frame_sampling=self.frame_sampling,
                depth_filter=self.depth_filter,
                cache_size=self.cache_size,
                files_per_site=self.files_per_site,
                site_order=self.site_order,
                pad_missing_sites=self.pad_missing_sites,
                max_sites=self.max_sites
            )
            
            self.patient_test = PatientLevelDataset(
                root_dir=self.root_dir,
                labels_csv=self.labels_csv,
                file_metadata_csv=self.file_metadata_csv,
                image_folder=self.image_folder,
                video_folder=self.video_folder,
                split='test',
                split_csv=self.split_csv,
                image_transforms=self.image_transforms,
                video_transforms=self.video_transforms,
                mode='video',
                depth_filter=self.depth_filter,
                frame_sampling=self.frame_sampling,
                cache_size=self.cache_size,
                files_per_site=self.files_per_site,
                site_order=self.site_order,
                pad_missing_sites=self.pad_missing_sites,
                max_sites=self.max_sites
            )
    
    def patient_level_dataloader(self, split='train'):
        if split == 'train':
            dataset = self.patient_train
            shuffle = True
        elif split == 'val':
            dataset = self.patient_val
            shuffle = False
        elif split == 'test':
            dataset = self.patient_test
            shuffle = False
        else:
            raise ValueError(f"Invalid split: {split}")
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_patient_batch, 
            prefetch_factor=2,
        )
    
    def train_dataloader(self, stage='patient_level'):
        if stage == 'patient_level':
            return self.patient_level_dataloader('train')
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def val_dataloader(self, stage='patient_level'):
        if stage == 'patient_level':
            return self.patient_level_dataloader('val')
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def test_dataloader(self):
        return self.patient_level_dataloader('test')