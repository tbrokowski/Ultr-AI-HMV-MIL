import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Union, Optional, Callable
from collections import OrderedDict
from functools import lru_cache

from ultr_ai.dataset.utils import NUM_PATH_CLASSES, SimpleVideoTransforms

class PatientLevelDataset(Dataset):
    
    def __init__(self, 
                 root_dir: str,
                 labels_csv: str,
                 file_metadata_csv: str,
                 image_folder: str = 'images',
                 video_folder: str = 'videos',
                 split: str = 'train',
                 split_csv: Optional[str] = None,
                 image_transforms: Optional[Callable] = None,
                 depth_filter: str = 'all',
                 video_transforms: Optional[Callable] = None,
                 mode: str = 'video',
                 frame_sampling: int = 16,
                 selected_sites: Optional[List[str]] = None,
                 cache_size: int = 100,
                 files_per_site: Optional[Union[int, str]] = 'all', 
                 site_order: Optional[List[str]] = None,   
                 pad_missing_sites: bool = True,           
                 max_sites: Optional[int] = None           
                 ):
        
        self.root_dir = root_dir
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.split = split
        self.depth_filter = depth_filter
        self.mode = mode.lower()

        
        self.files_per_site = files_per_site
        self.site_order = site_order
        self.pad_missing_sites = pad_missing_sites
        self.max_sites = max_sites
        self._uses_controlled_selection = (
            files_per_site != 'all' and pad_missing_sites
        )

        if self.mode not in ['video', 'image', 'both']:
            raise ValueError(f"Invalid mode: {mode}. Use 'video', 'image', or 'both'.")
        self.frame_sampling = frame_sampling
        self.selected_sites = selected_sites
        
        self.image_transforms = image_transforms or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.video_transforms = video_transforms or SimpleVideoTransforms()
        
        self.site_mapping = OrderedDict([
            ("<PAD>", 0),
            ("QAID", 1),
            ("QAIG", 2),
            ("QASD", 3),
            ("QASG", 4),
            ("QLD", 5),
            ("QLG", 6),
            ("QPID", 7),
            ("QPIG", 8),
            ("QPSD", 9),
            ("QPSG", 10),
            ("APXD", 11),
            ("APXG", 12),
            ("QSLD", 13), 
            ("QSLG", 14),
            ("SAD", 15), ("SLD", 16), ("SAG", 17), ("SLG", 18),
            ("SPD", 19), ("SPG", 20)
        ])

        self.labels_df = pd.read_csv(labels_csv)
        self.file_metadata_df = pd.read_csv(file_metadata_csv)

        required_labels = ['TB Label', 'Pneumonia', 'Covid']
        for label in required_labels:
            if label not in self.labels_df.columns:
                available_columns = list(self.labels_df.columns)
                raise ValueError(f"Label '{label}' not found in labels CSV. Available columns: {available_columns}")

        self.labels_df['patient_id'] = self.labels_df['record_id'].astype(str)
        self.file_metadata_df['patient_id'] = self.file_metadata_df['Patient ID'].astype(str)

        self.labels_df['patient_id_padded'] = self.labels_df['record_id'].astype(str).str.zfill(3)
        self.file_metadata_df['patient_id_padded'] = self.file_metadata_df['Patient ID'].astype(str).str.zfill(3)

        label_patients = set(self.labels_df['patient_id'].unique())
        label_patients_padded = set(self.labels_df['patient_id_padded'].unique())
        file_patients = set(self.file_metadata_df['patient_id'].unique())
        file_patients_padded = set(self.file_metadata_df['patient_id_padded'].unique())
        
        matches_unpadded = len(label_patients.intersection(file_patients))
        matches_padded = len(label_patients_padded.intersection(file_patients_padded))
        
        if matches_padded > matches_unpadded:
            self.labels_df['patient_id'] = self.labels_df['patient_id_padded']
            self.file_metadata_df['patient_id'] = self.file_metadata_df['patient_id_padded']

        if split_csv and split != 'all':
            self._load_split_info(split_csv)
            self._filter_by_split()

        self._apply_depth_filter()

        self._extract_site_labels()
        
        if self.selected_sites:
            self.file_metadata_df = self.file_metadata_df[
                self.file_metadata_df['Site'].isin(self.selected_sites)
            ]
        
        self._index_files()
        
        self.patients = self._group_by_patient()
        
        self.video_cache = lru_cache(maxsize=cache_size)(self._load_video_uncached)
        self.image_cache = lru_cache(maxsize=cache_size)(self._load_image)
    
    def _load_split_info(self, split_csv):
        split_df = pd.read_csv(split_csv, dtype=str).fillna("")
        
        if 'train_ids' in split_df.columns and 'test_ids' in split_df.columns:
            train_ids = split_df['train_ids'].dropna().astype(str).tolist()
            test_ids = split_df['test_ids'].dropna().astype(str).tolist()
            val_ids = []
            if 'valid_ids' in split_df.columns:
                val_ids = split_df['valid_ids'].dropna().astype(str).tolist()
            
            self.patient_splits = {}
            for pid in train_ids:
                self.patient_splits[pid] = 'train'
            for pid in test_ids:
                self.patient_splits[pid] = 'test'
            for pid in val_ids:
                self.patient_splits[pid] = 'val'
        else:
            self.patient_splits = dict(zip(split_df['patient_id'].astype(str), split_df['split']))
        
        self.file_metadata_df['split'] = self.file_metadata_df['patient_id'].map(
            lambda pid: self.patient_splits.get(pid, 'unknown')
        )
    
    def _filter_by_split(self):
        split_value = self.split
        if split_value == 'val' and not any(self.file_metadata_df['split'] == 'val'):
            split_value = 'valid'
        elif split_value == 'valid' and not any(self.file_metadata_df['split'] == 'valid'):
            split_value = 'val'
            
        self.file_metadata_df = self.file_metadata_df[
            self.file_metadata_df['split'] == split_value
        ]

    def _apply_depth_filter(self):
        if self.depth_filter == 'all':
            pass
        elif self.depth_filter == '5':
            self.file_metadata_df = self.file_metadata_df[
                self.file_metadata_df['Depth'].astype(int) < 10
            ]
        elif self.depth_filter == '15':
            self.file_metadata_df = self.file_metadata_df[
                self.file_metadata_df['Depth'].astype(int) > 10
            ]
        else:
            raise ValueError(f"Invalid depth_filter: {self.depth_filter}. Use 'all', '5', or '15'.")
    
    def _extract_site_labels(self):
        
        self.site_codes = set()
        for col in self.labels_df.columns:
            if '_' in col and col not in ['record_id', 'patient_id', 'patient_id_padded', 'TB Label', 'Pneumonia', 'Covid']:
                site = col.split('_')[0]
                if site == 'QLID':
                    site = 'QLD'
                elif site == 'QLIG':
                    site = 'QLG'
                elif site == 'QSLD':
                    site = 'QSLD'
                elif site == 'QSLG':
                    site = 'QSLG'
                self.site_codes.add(site)
        
        self.finding_columns = [col for col in self.labels_df.columns 
                              if not col.endswith('_severity') 
                              and col not in ['record_id', 'patient_id', 'patient_id_padded', 'TB Label', 'Pneumonia', 'Covid']
                              and '_' in col]
        
        self.patient_labels = {}
        for _, row in self.labels_df.iterrows():
            patient_id = row['patient_id']
            self.patient_labels[patient_id] = {
                'TB Label': row.get('TB Label', -1),
                'Pneumonia Label': row.get('Pneumonia', -1),
                'Covid Label': row.get('Covid', -1)
            }
        
        for label_name in ['TB Label', 'Pneumonia', 'Covid']:
            self.file_metadata_df[label_name] = self.file_metadata_df['patient_id'].map(
                lambda pid: self.patient_labels.get(pid, {}).get(label_name, -1)
            )
        
        self.site_labels = {}
        
        for _, row in self.labels_df.iterrows():
            patient_id = row['patient_id']
            
            if patient_id not in self.site_labels:
                self.site_labels[patient_id] = {}
            
            for site in self.site_codes:
                if site not in self.site_labels[patient_id]:
                    self.site_labels[patient_id][site] = {'findings': {}}
                
                self.site_labels[patient_id][site]['findings'] = {}
                
                site_prefixes = [site]
                if site == 'QLD':
                    site_prefixes.append('QLID')
                elif site == 'QLG':
                    site_prefixes.append('QLIG')
                
                for prefix in site_prefixes:
                    for col in self.finding_columns:
                        if col.startswith(f"{prefix}_") and not pd.isna(row[col]) and row[col] != -1:
                            finding_type = col[len(prefix)+1:]
                            
                            if finding_type == 'A-line' and row[col] == 1:
                                mapped_value = 1
                            elif finding_type == 'B-lines' and row[col] == 1:
                                mapped_value = 1
                            elif finding_type == 'Confluent B-lines' and row[col] == 1:
                                mapped_value = 1
                            elif finding_type == 'small Consolidations or Nodules' and row[col] == 1:
                                mapped_value = 1
                            elif finding_type == 'large Consolidations' and row[col] == 1:
                                mapped_value = 1
                            elif finding_type == 'Pleural effusion' and row[col] == 1:
                                mapped_value = 1
                            else:
                                mapped_value = -1
                                
                            if mapped_value != -1:
                                self.site_labels[patient_id][site]['findings'][finding_type] = mapped_value
        
        self.file_metadata_df['site_findings'] = self.file_metadata_df.apply(
            lambda row: self._get_site_findings(row['patient_id'], row['Site']), axis=1
        )
    
    def _get_site_findings(self, patient_id, site):
        mapped_site = site
        if site == 'QLID':
            mapped_site = 'QLD'
        elif site == 'QLIG':
            mapped_site = 'QLG'
        
        if patient_id in self.site_labels and mapped_site in self.site_labels[patient_id]:
            return self.site_labels[patient_id][mapped_site].get('findings', {})
        return {}
    
    def _index_files(self):
        self.image_paths = {}
        self.video_paths = {}
        
        if self.mode in ['image', 'both']:
            images_dir = os.path.join(self.root_dir, self.image_folder)
            if os.path.exists(images_dir):
                for img_file in glob.glob(os.path.join(images_dir, "*.png")):
                    filename = os.path.basename(img_file)
                    file_key = os.path.splitext(filename)[0]
                    self.image_paths[file_key] = img_file
        
        if self.mode in ['video', 'both']:
            videos_dir = os.path.join(self.root_dir, self.video_folder)
            if os.path.exists(videos_dir):
                for vid_file in glob.glob(os.path.join(videos_dir, "*.mp4")):
                    filename = os.path.basename(vid_file)
                    file_key = os.path.splitext(filename)[0]
                    self.video_paths[file_key] = vid_file
        
        self.file_metadata_df['file_key'] = self.file_metadata_df.apply(
            lambda row: f"{row['Patient ID']}_{row['Site']}_{row['Depth']}_{row['Count']}", 
            axis=1
        )
    
    def _group_by_patient(self):
        patients = {}
        
        if self.site_order is not None:
            ordered_sites = self.site_order
        else:
            ordered_sites = [site for site in self.site_mapping.keys() if site != "<PAD>"]
        
        if self.max_sites is not None:
            ordered_sites = ordered_sites[:self.max_sites]
        
        patient_groups = self.file_metadata_df.groupby('patient_id')
        
        for patient_id, group in patient_groups:
            patient_labels = self.patient_labels.get(patient_id, {})
            if not patient_labels or all(label == -1 for label in patient_labels.values()):
                continue
            
            if patient_id not in patients:
                patients[patient_id] = {
                    'patient_labels': patient_labels,
                    'files': []
                }
            
            patient_files_by_site = {}
            for _, file_info in group.iterrows():
                file_key = file_info['file_key']
                site = file_info['Site']
                
                file_exists = False
                if (self.mode == 'video' and file_key in self.video_paths) or \
                   (self.mode == 'image' and file_key in self.image_paths) or \
                   (self.mode == 'both' and (file_key in self.video_paths or file_key in self.image_paths)):
                    file_exists = True
                
                if file_exists:
                    if site not in patient_files_by_site:
                        patient_files_by_site[site] = []
                    
                    depth = int(file_info['Depth'])
                    site_findings = file_info.get('site_findings', {})
                    
                    mapped_site = site
                    if site == 'QLID':
                        mapped_site = 'QLD'
                    elif site == 'QLIG':
                        mapped_site = 'QLG'
                    
                    patient_files_by_site[site].append({
                        'file_key': file_key,
                        'site': site,
                        'site_index': self.site_mapping.get(mapped_site, 0),
                        'depth': depth,
                        'site_findings': site_findings,
                        'has_video': file_key in self.video_paths,
                        'has_image': file_key in self.image_paths,
                        'is_real': True
                    })
            
            selected_files = []
            
            for site in ordered_sites:
                possible_sites = [site]
                if site == 'QLD':
                    possible_sites.append('QLID')
                elif site == 'QLG':
                    possible_sites.append('QLIG')
                
                site_files = []
                for possible_site in possible_sites:
                    if possible_site in patient_files_by_site:
                        site_files.extend(patient_files_by_site[possible_site])
                
                if site_files:
                    if self.files_per_site == 'all':
                        selected_files.extend(site_files)
                    else:
                        site_files.sort(key=lambda x: x['depth'])
                        actual_files = site_files[:self.files_per_site]
                        selected_files.extend(actual_files)
                        
                        num_padding_needed = self.files_per_site - len(actual_files)
                        for pad_idx in range(num_padding_needed):
                            selected_files.append({
                                'file_key': f"PAD_{site}_{len(actual_files) + pad_idx}",
                                'site': site,
                                'site_index': self.site_mapping.get(site, 0),
                                'depth': -1,
                                'site_findings': {},
                                'has_video': False,
                                'has_image': False,
                                'is_real': False
                            })
                
                elif self.pad_missing_sites and self.files_per_site != 'all':
                    for pad_idx in range(self.files_per_site):
                        selected_files.append({
                            'file_key': f"PAD_{site}_{pad_idx}",
                            'site': site,
                            'site_index': self.site_mapping.get(site, 0),
                            'depth': -1,
                            'site_findings': {},
                            'has_video': False,
                            'has_image': False,
                            'is_real': False
                        })
                        
            patients[patient_id]['files'] = selected_files
        
        patients = {pid: data for pid, data in patients.items() if data['files']}
        
        patient_samples = [
            {'patient_id': pid, **data}
            for pid, data in patients.items()
        ]
        
        print(f"Created dataset with {len(patient_samples)} patients for {self.split} split")
        if self.files_per_site != 'all':
            print(f"Using {self.files_per_site} files per site")
        if self.site_order:
            print(f"Using custom site order: {self.site_order}")
        print(f"Padding missing sites: {self.pad_missing_sites}")
        
        return patient_samples
    
    def _load_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            if self.image_transforms:
                image = self.image_transforms(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def _load_video_uncached(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return torch.empty(0, 3, 224, 224)
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count <= 0:
                print(f"Error: Video {video_path} has no frames")
                return torch.empty(0, 3, 224, 224)
            
            indices = np.linspace(0, frame_count - 1, self.frame_sampling, dtype=int)
            frames = []
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frames.append(frame)
            
            cap.release()
            
            if not frames:
                return torch.empty(0, 3, 224, 224)
            
            if self.video_transforms:
                video_tensor = self.video_transforms(frames)
            else:
                simple_transforms = SimpleVideoTransforms()
                video_tensor = simple_transforms(frames)
            
            return video_tensor
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return torch.empty(0, 3, 224, 224)
    
    def _get_findings_onehot(self, findings_dict, num_classes=NUM_PATH_CLASSES):
        one_hot = torch.zeros(num_classes)
        
        if not findings_dict or not isinstance(findings_dict, dict):
            return one_hot
        
        if 'A-line' in findings_dict and findings_dict['A-line'] == 1:
            one_hot[0] = 1
        
        if 'large Consolidations' in findings_dict and findings_dict['large Consolidations'] == 1:
            one_hot[1] = 1

        if ('Pleural effusion' in findings_dict and findings_dict['Pleural effusion'] == 1):
            one_hot[2] = 1
        
        other_conditions = [
            ('B-lines' in findings_dict and findings_dict['B-lines'] == 1),
            ('Confluent B-lines' in findings_dict and findings_dict['Confluent B-lines'] == 1),
            ('small Consolidations or Nodules' in findings_dict and findings_dict['small Consolidations or Nodules'] == 1),
        ]
        
        if any(other_conditions):
            one_hot[3] = 1
        
        return one_hot

    def create_site_order_presets(self):
        presets = {}
        
        presets.update({
            'anatomical': ['QASD', 'QAID', 'QASG', 'QAIG', 'QLD', 'QLG', 'QPSD', 'QPID', 'QPSG', 'QPIG', 'APXD', 'APXG', 'QSLD', 'QSLG'],
            'anterior_first': ['QASD', 'QASG', 'QAID', 'QAIG', 'QLD', 'QLG', 'QPSD', 'QPSG', 'QPID', 'QPIG'],
            'bilateral_pairs': ['QASD', 'QASG', 'QAID', 'QAIG', 'QLD', 'QLG', 'QPSD', 'QPSG', 'QPID', 'QPIG'],
            'standard_first': ['QASD', 'QAID', 'QLD', 'QASG', 'QAIG', 'QLG', 'QPSD', 'QPID', 'QPSG', 'QPIG'],
            'sweep_first': ['SAD', 'SLD', 'SAG', 'SLG', 'SPD', 'SPG', 'QASD', 'QAID', 'QLD', 'QASG', 'QAIG', 'QLG', 'QPSD', 'QPID', 'QPSG', 'QPIG'],
            'anterior_posterior': ['QASD', 'QAID', 'SAD', 'QASG', 'QAIG', 'SAG', 'QLD', 'SLD', 'QLG', 'SLG', 'QPSD', 'QPID', 'SPD', 'QPSG', 'QPIG', 'SPG'],
            'sweep_only': ['SAD', 'SLD', 'SAG', 'SLG', 'SPD', 'SPG'],
            'standard_only': ['QASD', 'QAID', 'QLD', 'QASG', 'QAIG', 'QLG', 'QPSD', 'QPID', 'QPSG', 'QPIG']
        })
        
        return presets

    def get_site_statistics(self):
        stats = {
            'total_patients': len(self.patients),
            'site_coverage': {},
            'files_per_patient': [],
            'real_files_per_patient': []
        }
        
        for site in self.site_mapping.keys():
            if site != "<PAD>":
                stats['site_coverage'][site] = {
                    'patients_with_site': 0,
                    'total_files': 0,
                    'real_files': 0,
                    'padded_files': 0
                }
        
        for patient in self.patients:
            files = patient['files']
            stats['files_per_patient'].append(len(files))
            
            real_files = sum(1 for f in files if f['is_real'])
            stats['real_files_per_patient'].append(real_files)
            
            patient_sites = set()
            for file_info in files:
                site = file_info['site']
                mapped_site = site
                if site == 'QLID':
                    mapped_site = 'QLD'
                elif site == 'QLIG':
                    mapped_site = 'QLG'
                    
                if mapped_site in stats['site_coverage']:
                    stats['site_coverage'][mapped_site]['total_files'] += 1
                    if file_info['is_real']:
                        stats['site_coverage'][mapped_site]['real_files'] += 1
                        patient_sites.add(mapped_site)
                    else:
                        stats['site_coverage'][mapped_site]['padded_files'] += 1
            
            for site in patient_sites:
                stats['site_coverage'][site]['patients_with_site'] += 1
        
        return stats

    def print_dataset_summary(self):
        print("=== Dataset Configuration ===")
        print(f"Multi-task labels: TB Label, Pneumonia Label, Covid Label")
        print(f"Files per site: {self.files_per_site}")
        print(f"Pad missing sites: {self.pad_missing_sites}")
        print(f"Max sites: {self.max_sites}")
        if self.site_order:
            print(f"Custom site order: {self.site_order}")
        
        stats = self.get_site_statistics()
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Total patients: {stats['total_patients']}")
        if stats['files_per_patient']:
            print(f"Average files per patient: {np.mean(stats['files_per_patient']):.1f}")
            print(f"Average real files per patient: {np.mean(stats['real_files_per_patient']):.1f}")
        
        print(f"\n=== Site Coverage ===")
        for site, coverage in stats['site_coverage'].items():
            if coverage['total_files'] > 0:
                coverage_pct = (coverage['patients_with_site'] / stats['total_patients']) * 100
                print(f"{site:6s}: {coverage['patients_with_site']:3d}/{stats['total_patients']:3d} patients ({coverage_pct:5.1f}%), "
                      f"{coverage['real_files']:4d} real files, {coverage['padded_files']:4d} padded")
    
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient = self.patients[idx]
        patient_id = patient['patient_id']
        patient_labels = patient['patient_labels']
        files = patient['files']
        
        site_data = []
        
        for file_info in files:
            file_key = file_info['file_key']
            site = file_info['site']
            site_index = file_info['site_index']
            depth = file_info['depth']
            is_real = file_info['is_real']
            site_findings = file_info['site_findings']
            
            findings_onehot = self._get_findings_onehot(site_findings)
            
            video = None
            image = None
            
            if is_real:
                has_video = file_info['has_video']
                has_image = file_info['has_image']
                
                if has_video and self.mode in ['video', 'both']:
                    video_path = self.video_paths[file_key]
                    video = self.video_cache(video_path)
                    
                    if video is None or video.numel() == 0:
                        is_real = False
                
                if has_image and self.mode in ['image', 'both']:
                    image_path = self.image_paths[file_key]
                    image = self.image_cache(image_path)
                    
                    if image is None:
                        is_real = False
            
            if not is_real:
                if self.mode in ['video', 'both']:
                    video = torch.zeros(self.frame_sampling, 3, 224, 224)
                if self.mode in ['image', 'both']:
                    image = torch.zeros(3, 224, 224)
                findings_onehot = torch.zeros(NUM_PATH_CLASSES)
            
            site_data.append({
                'file_key': file_key,
                'site': site,
                'site_index': site_index,
                'depth': depth,
                'findings_onehot': findings_onehot,
                'site_findings': site_findings,
                'video': video,
                'image': image,
                'is_real': is_real
            })
        
        if not site_data:
            return {
                'patient_id': patient_id,
                'tb_label': patient_labels.get('TB Label', -1),
                'pneumonia_label': patient_labels.get('Pneumonia Label', -1),
                'covid_label': patient_labels.get('Covid Label', -1),
                'num_sites': 0,
                'site_indices': torch.zeros(1, dtype=torch.long),
                'site_videos': torch.zeros(1, self.frame_sampling, 3, 224, 224),
                'site_images': torch.zeros(1, 3, 224, 224),
                'site_findings': torch.zeros(1, NUM_PATH_CLASSES),
                'is_real_mask': torch.zeros(1, dtype=torch.bool),
                'is_valid': False,
                '_uses_controlled_selection': self._uses_controlled_selection
            }
        
        num_sites = len(site_data)
        site_indices = torch.tensor([s['site_index'] for s in site_data], dtype=torch.long)
        site_findings = torch.stack([s['findings_onehot'] for s in site_data])
        is_real_mask = torch.tensor([s['is_real'] for s in site_data], dtype=torch.bool)
        
        if self.mode in ['video', 'both']:
            videos = [s['video'] for s in site_data if s['video'] is not None]
            if videos and all(v.shape[0] == videos[0].shape[0] for v in videos):
                site_videos = torch.stack(videos)
            else:
                max_frames = max(v.shape[0] for v in videos) if videos else self.frame_sampling
                C, H, W = (3, 224, 224)
                site_videos = torch.zeros(num_sites, max_frames, C, H, W)
                for i, video in enumerate(videos):
                    if video is not None:
                        site_videos[i, :video.shape[0]] = video
        else:
            site_videos = torch.zeros(num_sites, self.frame_sampling, 3, 224, 224)
        
        if self.mode in ['image', 'both']:
            images = [s['image'] for s in site_data if s['image'] is not None]
            if images:
                site_images = torch.stack(images)
            else:
                site_images = torch.zeros(num_sites, 3, 224, 224)
        else:
            site_images = torch.zeros(num_sites, 3, 224, 224)
        
        result = {
            'patient_id': patient_id,
            'tb_label': patient_labels.get('TB Label', -1),
            'pneumonia_label': patient_labels.get('Pneumonia Label', -1),
            'covid_label': patient_labels.get('Covid Label', -1),
            'num_sites': num_sites,
            'site_indices': site_indices,
            'site_videos': site_videos,
            'site_images': site_images,
            'site_findings': site_findings,
            'is_real_mask': is_real_mask,
            'is_valid': True,
            '_uses_controlled_selection': self._uses_controlled_selection
        }
        
        return result