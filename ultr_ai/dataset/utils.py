import random
import torch
import torchvision.transforms.functional as F
from PIL import ImageEnhance, ImageFilter

NUM_PATH_CLASSES = 4

class UltrasoundPreprocessing(object):
    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        return img

class UltrasoundNoiseAugment(object):
    def __call__(self, tensor):
        speckle = torch.randn_like(tensor) * 0.2
        tensor = tensor * (1 + speckle)
        tensor = torch.clamp(tensor, 0, 1)
        return tensor

class TemporallyConsistentTransforms:
    def __init__(self, 
                 resize_size=(224, 224),
                 degrees=25,
                 translate=(0.15, 0.15),
                 scale=(0.65, 1.45),
                 brightness=0.3,
                 contrast=0.3,
                 blur_kernel_size=3,
                 blur_sigma=(0.1, 0.5),
                 noise_std=0.2,
                 augment_prob=0.5,
                 blur_prob=0.2,
                 mean=[0.45, 0.45, 0.45],
                 std=[0.225, 0.225, 0.225]):
        
        self.resize_size = resize_size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.brightness = brightness
        self.contrast = contrast
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std
        self.augment_prob = augment_prob
        self.blur_prob = blur_prob
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __call__(self, video_frames):
        if not video_frames:
            return torch.empty(0, 3, *self.resize_size)
        
        augment_params = self._sample_augmentation_parameters()
        
        transformed_frames = []
        for frame in video_frames:
            frame = self._apply_resize(frame)
            frame = self._apply_contrast_enhancement(frame, augment_params)
            frame = self._apply_affine_transform(frame, augment_params)
            frame = self._apply_color_jitter(frame, augment_params)
            frame = self._apply_gaussian_blur(frame, augment_params)
            
            frame_tensor = F.to_tensor(frame)
            transformed_frames.append(frame_tensor)
        
        video_tensor = torch.stack(transformed_frames)
        video_tensor = self._apply_noise(video_tensor, augment_params)
        video_tensor = self._apply_normalization(video_tensor)
        
        return video_tensor
    
    def _sample_augmentation_parameters(self):
        params = {}
        
        if random.random() < self.augment_prob:
            params['apply_affine'] = True
            params['angle'] = random.uniform(-self.degrees, self.degrees)
            params['translate'] = (
                random.uniform(-self.translate[0], self.translate[0]),
                random.uniform(-self.translate[1], self.translate[1])
            )
            params['scale'] = random.uniform(self.scale[0], self.scale[1])
        else:
            params['apply_affine'] = False
        
        params['brightness_factor'] = random.uniform(
            max(0, 1 - self.brightness), 1 + self.brightness
        )
        params['contrast_factor'] = random.uniform(
            max(0, 1 - self.contrast), 1 + self.contrast
        )
        
        if random.random() < self.blur_prob:
            params['apply_blur'] = True
            params['blur_sigma'] = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
        else:
            params['apply_blur'] = False
        
        params['noise_multiplier'] = torch.randn(1, 1, 1, 1) * self.noise_std
        
        return params
    
    def _apply_resize(self, frame):
        return F.resize(frame, self.resize_size)
    
    def _apply_contrast_enhancement(self, frame, params):
        enhancer = ImageEnhance.Contrast(frame)
        return enhancer.enhance(1.2)
    
    def _apply_affine_transform(self, frame, params):
        if not params['apply_affine']:
            return frame
        
        width, height = frame.size
        translate_pixels = (
            int(params['translate'][0] * width),
            int(params['translate'][1] * height)
        )
        
        return F.affine(
            frame,
            angle=params['angle'],
            translate=translate_pixels,
            scale=params['scale'],
            shear=0,
            fill=0
        )
    
    def _apply_color_jitter(self, frame, params):
        frame = F.adjust_brightness(frame, params['brightness_factor'])
        frame = F.adjust_contrast(frame, params['contrast_factor'])
        return frame
    
    def _apply_gaussian_blur(self, frame, params):
        if not params['apply_blur']:
            return frame
        
        return frame.filter(ImageFilter.GaussianBlur(radius=params['blur_sigma']))
    
    def _apply_noise(self, video_tensor, params):
        speckle = params['noise_multiplier'] * torch.randn_like(video_tensor)
        video_tensor = video_tensor * (1 + speckle)
        return torch.clamp(video_tensor, 0, 1)
    
    def _apply_normalization(self, video_tensor):
        return (video_tensor - self.mean) / self.std

class SimpleVideoTransforms:
    def __init__(self, 
                 resize_size=(224, 224),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        
        self.resize_size = resize_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __call__(self, video_frames):
        if not video_frames:
            return torch.empty(0, 3, *self.resize_size)
        
        transformed_frames = []
        for frame in video_frames:
            frame = F.resize(frame, self.resize_size)
            frame_tensor = F.to_tensor(frame)
            transformed_frames.append(frame_tensor)
        
        video_tensor = torch.stack(transformed_frames)
        video_tensor = (video_tensor - self.mean) / self.std
        
        return video_tensor
    
def collate_patient_batch(batch):
    
    valid_batch = [sample for sample in batch if sample['is_valid']]
    
    if not valid_batch:
        return {
            'patient_ids': [],
            'tb_labels': torch.zeros(0, dtype=torch.long),
            'pneumonia_labels': torch.zeros(0, dtype=torch.long),
            'covid_labels': torch.zeros(0, dtype=torch.long),
            'site_indices': torch.zeros(0, dtype=torch.long),
            'site_counts': torch.zeros(0, dtype=torch.long),
            'site_videos': torch.zeros(0, 0, 0, 3, 224, 224),
            'site_images': torch.zeros(0, 0, 3, 224, 224),
            'site_findings': torch.zeros(0, 0, NUM_PATH_CLASSES),
            'site_masks': torch.zeros(0, 0, dtype=torch.bool),
            '_mask_type': 'empty'
        }
    
    uses_controlled_selection = valid_batch[0].get('_uses_controlled_selection', False)
    
    batch_size = len(valid_batch)
    patient_ids = [sample['patient_id'] for sample in valid_batch]
    tb_labels = torch.tensor([sample['tb_label'] for sample in valid_batch], dtype=torch.long)
    pneumonia_labels = torch.tensor([sample['pneumonia_label'] for sample in valid_batch], dtype=torch.long)
    covid_labels = torch.tensor([sample['covid_label'] for sample in valid_batch], dtype=torch.long)
    
    max_sites = max(sample['num_sites'] for sample in valid_batch)
    
    site_counts = torch.tensor([sample['num_sites'] for sample in valid_batch], dtype=torch.long)
    site_indices = torch.zeros(batch_size, max_sites, dtype=torch.long)
    site_findings = torch.zeros(batch_size, max_sites, NUM_PATH_CLASSES)
    
    batch_padding_masks = torch.zeros(batch_size, max_sites, dtype=torch.bool)
    real_data_masks = torch.zeros(batch_size, max_sites, dtype=torch.bool)
    
    video_shape = valid_batch[0]['site_videos'].shape[1:]
    image_shape = valid_batch[0]['site_images'].shape[1:]
    
    site_videos = torch.zeros(batch_size, max_sites, *video_shape)
    site_images = torch.zeros(batch_size, max_sites, *image_shape)
    
    for i, sample in enumerate(valid_batch):
        num_sites = sample['num_sites']
        site_indices[i, :num_sites] = sample['site_indices']
        site_findings[i, :num_sites] = sample['site_findings']
        site_videos[i, :num_sites] = sample['site_videos']
        site_images[i, :num_sites] = sample['site_images']
        
        batch_padding_masks[i, :num_sites] = True
        
        real_data_masks[i, :num_sites] = sample['is_real_mask']
    
    if uses_controlled_selection:
        adaptive_site_masks = real_data_masks
        mask_type = 'real_data'
    else:
        adaptive_site_masks = batch_padding_masks  
        mask_type = 'batch_padding'
    
    result = {
        'patient_ids': patient_ids,
        'tb_labels': tb_labels,
        'pneumonia_labels': pneumonia_labels,
        'covid_labels': covid_labels,
        'site_indices': site_indices,
        'site_counts': site_counts,
        'site_videos': site_videos,
        'site_images': site_images,
        'site_findings': site_findings,
        'site_masks': adaptive_site_masks,
        
        'batch_padding_masks': batch_padding_masks,
        'real_data_masks': real_data_masks,
        '_mask_type': mask_type
    }
    
    return result

def analyze_batch_masks(batch):
    site_masks = batch['site_masks']
    if 'real_data_masks' in batch:
        real_data_masks = batch['real_data_masks']
        
        print("=== Batch Mask Analysis ===")
        print(f"Batch size: {site_masks.shape[0]}")
        print(f"Max sites per patient: {site_masks.shape[1]}")
        print(f"Mask type: {batch.get('_mask_type', 'unknown')}")
        
        for i in range(min(3, site_masks.shape[0])):
            valid_sites = site_masks[i].sum().item()
            real_sites = real_data_masks[i].sum().item()
            padded_sites = valid_sites - real_sites
            
            print(f"Patient {i}: {valid_sites} valid sites, {real_sites} real, {padded_sites} padded")
        
        total_valid = site_masks.sum().item()
        total_real = real_data_masks.sum().item()
        total_padded = total_valid - total_real
        
        print(f"Total: {total_valid} valid, {total_real} real, {total_padded} padded")
        if total_valid > 0:
            print(f"Real data ratio: {total_real/total_valid:.2%}")
    else:
        print("=== Basic Mask Analysis ===")
        print(f"Batch size: {site_masks.shape[0]}")
        print(f"Max sites per patient: {site_masks.shape[1]}")
        print(f"Total valid sites: {site_masks.sum().item()}")

def analyze_batch_labels(batch):
    print("=== Multi-Task Label Analysis ===")
    print(f"Batch size: {len(batch['patient_ids'])}")
    
    for task_name, labels in [('TB', batch['tb_labels']), 
                             ('Pneumonia', batch['pneumonia_labels']), 
                             ('Covid', batch['covid_labels'])]:
        valid_labels = labels[labels != -1]
        if len(valid_labels) > 0:
            positive_count = (valid_labels == 1).sum().item()
            negative_count = (valid_labels == 0).sum().item()
            print(f"{task_name}: {len(valid_labels)} valid labels, {positive_count} positive, {negative_count} negative")
        else:
            print(f"{task_name}: No valid labels in batch")