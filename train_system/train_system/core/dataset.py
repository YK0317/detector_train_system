"""
Unified Dataset for Train System

Handles different data formats and sources for the training system.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from PIL import Image
import json
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import os

from ..config import DataConfig


class UnifiedDataset(Dataset):
    """
    Unified dataset class that can handle different data formats
    """
    
    def __init__(self, config: DataConfig, split: str = 'train', transform=None):
        """
        Initialize unified dataset
        
        Args:
            config: Data configuration
            split: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.samples = []
        
        # Load samples based on data type
        if config.type == "folder":
            self._load_folder_data(split)
        elif config.type == "class_folders":
            self._load_class_folders_data(split)
        elif config.type == "csv":
            self._load_csv_data(split)
        elif config.type == "json":
            self._load_json_data(split)
        else:
            raise ValueError(f"Unsupported data type: {config.type}")
        
        # Apply sample limit if specified
        if config.max_samples and len(self.samples) > config.max_samples:
            self.samples = self.samples[:config.max_samples]
        
        self._log_dataset_info()
    
    def _load_folder_data(self, split: str):
        """Load data from folder structure"""
        if split == 'train':
            data_path = Path(self.config.train_path)
        elif split == 'val':
            data_path = Path(self.config.val_path)
        elif split == 'test' and self.config.test_path:
            data_path = Path(self.config.test_path)
        else:
            raise ValueError(f"No path specified for split: {split}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Load images from class folders
        for class_name, class_idx in self.config.class_mapping.items():
            class_dir = data_path / class_name
            if class_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    for img_path in class_dir.glob(ext):
                        self.samples.append((str(img_path), class_idx))
    
    def _load_class_folders_data(self, split: str):
        """Load data from separate class folders"""
        if split == 'train':
            # For training, use all specified class folders
            class_paths = {}
            
            # Option 1: Use base train_path + relative real_path/fake_path
            if (hasattr(self.config, 'train_path') and self.config.train_path and 
                hasattr(self.config, 'real_path') and self.config.real_path and
                hasattr(self.config, 'fake_path') and self.config.fake_path):
                
                base_path = Path(self.config.train_path)
                class_paths['real'] = str(base_path / self.config.real_path)
                class_paths['fake'] = str(base_path / self.config.fake_path)
            
            # Option 2: Use absolute real_path and fake_path (backward compatibility)
            elif (hasattr(self.config, 'real_path') and self.config.real_path and
                  hasattr(self.config, 'fake_path') and self.config.fake_path and
                  (Path(self.config.real_path).is_absolute() or Path(self.config.fake_path).is_absolute())):
                class_paths['real'] = self.config.real_path
                class_paths['fake'] = self.config.fake_path
            
            # Option 3: Use class_folders dict
            elif hasattr(self.config, 'class_folders') and self.config.class_folders:
                # Build full paths from train_path + class folder names
                train_base = Path(self.config.train_path)
                for class_name, folder_name in self.config.class_folders.items():
                    class_paths[class_name] = str(train_base / folder_name)
            
            if not class_paths:
                raise ValueError("No class folders specified. Use either:\n"
                               "1. train_path + real_path/fake_path (relative)\n"
                               "2. real_path/fake_path (absolute paths)\n" 
                               "3. class_folders dictionary")
            
        elif split == 'val':
            # For validation, create val paths from train paths
            class_paths = {}
            
            # Option 1: Base path + relative paths
            if (hasattr(self.config, 'train_path') and self.config.train_path and 
                hasattr(self.config, 'real_path') and self.config.real_path and
                hasattr(self.config, 'fake_path') and self.config.fake_path):
                
                if hasattr(self.config, 'val_path') and self.config.val_path:
                    # Use explicit val_path as base
                    base_path = Path(self.config.val_path)
                    class_paths['real'] = str(base_path / self.config.real_path)
                    class_paths['fake'] = str(base_path / self.config.fake_path)
                else:
                    # Try to infer validation path from train path
                    train_base = Path(self.config.train_path).parent
                    val_base = train_base / 'validation'
                    if val_base.exists():
                        class_paths['real'] = str(val_base / self.config.real_path)
                        class_paths['fake'] = str(val_base / self.config.fake_path)
                    else:
                        logging.warning("No validation path found, using training paths")
                        base_path = Path(self.config.train_path)
                        class_paths['real'] = str(base_path / self.config.real_path)
                        class_paths['fake'] = str(base_path / self.config.fake_path)
            
            # Option 2: Absolute paths (backward compatibility)
            elif (hasattr(self.config, 'real_path') and self.config.real_path and
                  hasattr(self.config, 'fake_path') and self.config.fake_path):
                
                # Convert absolute training paths to validation paths
                real_base = Path(self.config.real_path).parent
                fake_base = Path(self.config.fake_path).parent
                
                # Check for validation folder structure
                if (real_base / 'validation' / 'real').exists():
                    class_paths['real'] = str(real_base / 'validation' / 'real')
                    class_paths['fake'] = str(fake_base / 'validation' / 'fake')
                elif self.config.val_path:
                    # Use general val_path with class subfolders
                    val_base = Path(self.config.val_path)
                    class_paths['real'] = str(val_base / 'real')
                    class_paths['fake'] = str(val_base / 'fake')
                else:
                    logging.warning("No validation paths found, using training paths")
                    class_paths['real'] = self.config.real_path
                    class_paths['fake'] = self.config.fake_path
            
            # Option 3: class_folders dict
            elif hasattr(self.config, 'class_folders') and self.config.class_folders:
                # Assume validation folders follow similar pattern
                val_base = Path(self.config.val_path) if hasattr(self.config, 'val_path') and self.config.val_path else None
                
                if val_base and val_base.exists():
                    # Use explicit val_path with class folder names
                    for class_name, folder_name in self.config.class_folders.items():
                        class_paths[class_name] = str(val_base / folder_name)
                else:
                    # Try to infer validation path from train_path
                    train_base = Path(self.config.train_path)
                    val_base = train_base.parent / 'validation'
                    
                    if val_base.exists():
                        for class_name, folder_name in self.config.class_folders.items():
                            class_paths[class_name] = str(val_base / folder_name)
                    else:
                        logging.warning("No validation folder found, validation split will be empty")
                        # Return empty class_paths to avoid errors
                        class_paths = {}
        
        else:  # test split
            if hasattr(self.config, 'test_path') and self.config.test_path:
                test_base = Path(self.config.test_path)
                if (hasattr(self.config, 'real_path') and self.config.real_path and
                    hasattr(self.config, 'fake_path') and self.config.fake_path):
                    # Use relative paths with test_path base
                    class_paths['real'] = str(test_base / self.config.real_path)
                    class_paths['fake'] = str(test_base / self.config.fake_path)
                else:
                    # Use class names as subfolders
                    class_paths = {class_name: str(test_base / class_name) 
                                 for class_name in self.config.class_mapping.keys()}
            else:
                raise ValueError(f"No test path specified for {split} split")
        
        # Load images from each class folder
        for class_name, folder_path in class_paths.items():
            class_dir = Path(folder_path)
            print(f"🔍 DEBUG: Checking class {class_name} at {class_dir}")
            if not class_dir.exists():
                print(f"❌ DEBUG: Class folder not found: {class_dir}")
                logging.warning(f"Class folder not found: {class_dir}")
                continue
            
            # Get class index
            class_idx = self.config.class_mapping.get(class_name, 0)
            print(f"✅ DEBUG: Class {class_name} -> index {class_idx}")
            
            # Load images
            image_count = 0
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                images_found = list(class_dir.glob(ext))
                if images_found:
                    print(f"📁 DEBUG: Found {len(images_found)} {ext} files in {class_name}")
                for img_path in images_found:
                    self.samples.append((str(img_path), class_idx))
                    image_count += 1
            
            print(f"✅ DEBUG: Loaded {image_count} images from {class_name}")
        
        print(f"🎯 DEBUG: Total samples loaded: {len(self.samples)}")
    
    def _load_csv_data(self, split: str):
        """Load data from CSV file"""
        csv_path = getattr(self.config, f'{split}_path', None)
        if not csv_path:
            raise ValueError(f"No CSV path specified for {split} split")
        
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Expect columns: 'image_path' and 'label'
        if 'image_path' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'image_path' and 'label' columns")
        
        # Convert labels to indices if they're strings
        if df['label'].dtype == 'object':
            unique_labels = sorted(df['label'].unique())
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            df['label_idx'] = df['label'].map(label_to_idx)
            
            # Update class mapping if not provided
            if not hasattr(self.config, 'class_mapping') or not self.config.class_mapping:
                self.config.class_mapping = {label: idx for label, idx in label_to_idx.items()}
        else:
            df['label_idx'] = df['label']
        
        # Load samples
        for _, row in df.iterrows():
            img_path = row['image_path']
            if not os.path.isabs(img_path):
                # Make relative paths relative to CSV file
                img_path = csv_path.parent / img_path
            self.samples.append((str(img_path), int(row['label_idx'])))
    
    def _load_json_data(self, split: str):
        """Load data from JSON file"""
        json_path = getattr(self.config, f'{split}_path', None)
        if not json_path:
            raise ValueError(f"No JSON path specified for {split} split")
        
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Expect format: {"samples": [{"image_path": "path", "label": 0}, ...]}
        if 'samples' not in data:
            raise ValueError("JSON must contain 'samples' key")
        
        for sample in data['samples']:
            if 'image_path' not in sample or 'label' not in sample:
                raise ValueError("Each sample must contain 'image_path' and 'label'")
            
            img_path = sample['image_path']
            if not os.path.isabs(img_path):
                # Make relative paths relative to JSON file
                img_path = json_path.parent / img_path
            
            self.samples.append((str(img_path), int(sample['label'])))
        
        # Update class mapping from JSON if provided
        if 'class_mapping' in data:
            self.config.class_mapping = data['class_mapping']
    
    def _log_dataset_info(self):
        """Log dataset information"""
        logging.info(f"Loaded {len(self.samples)} samples for {self.split} split")
        
        # Count samples per class
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        for class_idx, count in class_counts.items():
            if hasattr(self.config, 'class_mapping') and self.config.class_mapping:
                class_name = next((name for name, idx in self.config.class_mapping.items() if idx == class_idx), str(class_idx))
            else:
                class_name = str(class_idx)
            logging.info(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logging.warning(f"Error loading {img_path}: {e}")
            # Return dummy image
            dummy_image = Image.new('RGB', (self.config.img_size, self.config.img_size), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets"""
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self.samples)
        num_classes = len(class_counts)
        
        weights = []
        for class_idx in sorted(class_counts.keys()):
            weight = total_samples / (num_classes * class_counts[class_idx])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.2) -> Tuple['UnifiedDataset', 'UnifiedDataset']:
        """
        Split dataset into train and validation sets
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if abs(train_ratio + val_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio must equal 1.0")
        
        total_samples = len(self.samples)
        train_size = int(total_samples * train_ratio)
        
        # Shuffle samples for random split
        import random
        shuffled_samples = self.samples.copy()
        random.shuffle(shuffled_samples)
        
        # Create new datasets
        train_dataset = UnifiedDataset.__new__(UnifiedDataset)
        train_dataset.config = self.config
        train_dataset.split = 'train'
        train_dataset.transform = self.transform
        train_dataset.samples = shuffled_samples[:train_size]
        
        val_dataset = UnifiedDataset.__new__(UnifiedDataset)
        val_dataset.config = self.config
        val_dataset.split = 'val'
        val_dataset.transform = self.transform
        val_dataset.samples = shuffled_samples[train_size:]
        
        return train_dataset, val_dataset


class DatasetFactory:
    """Factory for creating datasets from different sources"""
    
    @staticmethod
    def from_folder(data_path: str, 
                   img_size: int = 224,
                   class_mapping: Optional[Dict[str, int]] = None,
                   **kwargs) -> UnifiedDataset:
        """
        Create dataset from folder structure
        
        Args:
            data_path: Path to data folder with class subdirectories
            img_size: Target image size
            class_mapping: Optional mapping of class names to indices
            **kwargs: Additional dataset arguments
        """
        from ..config import DataConfig
        
        data_path = Path(data_path)
        
        # Auto-detect class mapping if not provided
        if class_mapping is None:
            class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
            class_mapping = {d.name: idx for idx, d in enumerate(sorted(class_dirs))}
        
        config = DataConfig(
            type="folder",
            train_path=str(data_path),
            img_size=img_size,
            class_mapping=class_mapping,
            **kwargs
        )
        
        return UnifiedDataset(config, split='train')
    
    @staticmethod
    def from_csv(csv_path: str, 
                img_size: int = 224,
                **kwargs) -> UnifiedDataset:
        """
        Create dataset from CSV file
        
        Args:
            csv_path: Path to CSV file with 'image_path' and 'label' columns
            img_size: Target image size
            **kwargs: Additional dataset arguments
        """
        from ..config import DataConfig
        
        config = DataConfig(
            type="csv",
            train_path=str(csv_path),
            img_size=img_size,
            **kwargs
        )
        
        return UnifiedDataset(config, split='train')
    
    @staticmethod
    def from_json(json_path: str,
                 img_size: int = 224,
                 **kwargs) -> UnifiedDataset:
        """
        Create dataset from JSON file
        
        Args:
            json_path: Path to JSON file
            img_size: Target image size
            **kwargs: Additional dataset arguments
        """
        from ..config import DataConfig
        
        config = DataConfig(
            type="json",
            train_path=str(json_path),
            img_size=img_size,
            **kwargs
        )
        
        return UnifiedDataset(config, split='train')
