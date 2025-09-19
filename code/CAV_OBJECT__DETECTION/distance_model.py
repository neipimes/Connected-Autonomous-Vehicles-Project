"""
Distance Detection Model for Connected Automated Vehicle (CAV)
Inference module for loading and using two trained distance prediction models:
- One for class 0 detections
- One for all other class detections

Usage Documentation:

Put the following statement in the imports:
from distance_model import load_distance_models

Call load models at start after you've loaded the YOLOv5 models:
distance_predictor = load_distance_models()

To use:
if distance_predictor:
    distance = distance_predictor.predict_distance(class_num, x_center, y_center, width, height)

Alternatively, if you have a list of detections you can pass in a list of tuples to the predict_batch() function

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Union, List, Tuple, Optional

class DistanceModelGated(nn.Module):
    """Advanced approach using gating mechanism for distance prediction (for other classes)"""
    def __init__(self, num_classes=25, spatial_input_dim=6, class_embed_dim=8, 
                 hidden_dim=16, output_dim=1):
        super(DistanceModelGated, self).__init__()
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        
        # Class to gate weights
        self.gate_generator = nn.Sequential(
            nn.Linear(class_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # Gate values between 0 and 1
        )
        
        # Spatial feature processing
        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        class_idx = x[:, 0].long()
        spatial_features = x[:, 1:7]  # x_center, y_center, width, height, normalized_area, shape_difference
        
        # Generate gates from class information
        class_embed = self.class_embedding(class_idx)
        gates = self.gate_generator(class_embed)
        
        # Process spatial features
        spatial_processed = self.spatial_processor(spatial_features)
        
        # Apply class-dependent gating
        gated_features = spatial_processed * gates
        
        # Final processing
        output = self.final_layers(gated_features)
        return F.softplus(output)


class DistanceModelClass0(nn.Module):
    """Larger model for single class (class 0) distance prediction - doubled parameters"""
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=1):
        super(DistanceModelClass0, self).__init__()
        
        # Larger feedforward network with more layers and wider hidden dimensions
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Direct processing of spatial features (no class info needed for class 0 only model)
        output = self.layers(x)
        # Changed from F.relu to F.softplus for better gradient flow and no zero clipping
        return F.softplus(output) * 100


class CAVDualDistancePredictor:
    """
    Wrapper class for dual distance prediction models optimized for CAV deployment.
    Uses separate models for class 0 and other classes.
    """
    
    def __init__(self, class0_checkpoint_path: str = 'distance_model_checkpoint_class0_only.pth',
                 other_classes_checkpoint_path: str = 'distance_model_checkpoint_no_class0.pth',
                 device: Optional[str] = None):
        """
        Initialize the CAV Dual Distance Predictor.
        
        Args:
            class0_checkpoint_path: Path to the trained model checkpoint for class 0
            other_classes_checkpoint_path: Path to the trained model checkpoint for other classes
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)
        """
        self.class0_checkpoint_path = class0_checkpoint_path
        self.other_classes_checkpoint_path = other_classes_checkpoint_path
        self.class0_model = None
        self.other_classes_model = None
        self.device = self._set_device(device)
        self.class0_loaded = False
        self.other_classes_loaded = False
        self.class0_training_config = None
        self.other_classes_training_config = None
        
        # Image dimensions for normalized area calculation (adjust if different)
        self.IMAGE_WIDTH = 1.0
        self.IMAGE_HEIGHT = 1.0
        self.image_area = self.IMAGE_WIDTH * self.IMAGE_HEIGHT
        
    def _set_device(self, device: Optional[str]) -> torch.device:
        """Set the computation device"""
        if device is None:
            # Auto-detect best available device
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def load_models(self) -> bool:
        """
        Load both trained models from checkpoints.
        
        Returns:
            bool: True if both models loaded successfully, False otherwise
        """
        class0_success = self._load_single_model(
            self.class0_checkpoint_path, 
            'class0'
        )
        
        other_classes_success = self._load_single_model(
            self.other_classes_checkpoint_path, 
            'other_classes'
        )
        
        success = class0_success and other_classes_success
        
        if success:
            print(f"✓ Both distance models loaded successfully")
            print(f"  Device: {self.device}")
        else:
            print(f"✗ Failed to load one or both models")
            
        return success
    
    def _load_single_model(self, checkpoint_path: str, model_type: str) -> bool:
        """
        Load a single model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model_type: Either 'class0' or 'other_classes'
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(checkpoint_path):
                print(f"Error: Checkpoint file not found at {checkpoint_path}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialize appropriate model based on type
            if model_type == 'class0':
                model = DistanceModelClass0()
            else:
                model = DistanceModelGated()
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            # Store model and training configuration
            if model_type == 'class0':
                self.class0_model = model
                self.class0_training_config = checkpoint.get('training_config', {})
                self.class0_loaded = True
                print(f"✓ Class 0 model loaded successfully")
                print(f"  Training loss: {checkpoint['loss']:.4f}")
                if self.class0_training_config:
                    print(f"  Training samples: {self.class0_training_config.get('training_samples', 'N/A')}")
            else:
                self.other_classes_model = model
                self.other_classes_training_config = checkpoint.get('training_config', {})
                self.other_classes_loaded = True
                print(f"✓ Other classes model loaded successfully")
                print(f"  Training loss: {checkpoint['loss']:.4f}")
                if self.other_classes_training_config:
                    print(f"  Training samples: {self.other_classes_training_config.get('training_samples', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            return False
    
    def _get_model_for_class(self, class_num: Union[int, float]) -> Optional[torch.nn.Module]:
        """
        Get the appropriate model based on class number.
        
        Args:
            class_num: Object class number
            
        Returns:
            torch.nn.Module: The appropriate model, or None if not loaded
        """
        if class_num == 0:
            return self.class0_model if self.class0_loaded else None
        else:
            return self.other_classes_model if self.other_classes_loaded else None
    
    def _calculate_features_class0(self, x_center: float, y_center: float, 
                                 width: float, height: float) -> np.ndarray:
        """
        Calculate features for class 0 model (no class information needed).
        
        Args:
            x_center: Normalized x center coordinate
            y_center: Normalized y center coordinate  
            width: Normalized width
            height: Normalized height
            
        Returns:
            np.ndarray: Array of features [x_center, y_center, width, height, normalized_area, shape_difference]
        """
        # Calculate normalized area
        bounding_box_area = width * height
        normalized_area = bounding_box_area / self.image_area
        
        # Calculate shape difference (bounded between -1 and 1)
        epsilon = 1e-8  # Prevent division by zero
        shape_difference = (width - height) / (width + height + epsilon)
        
        return np.array([x_center, y_center, width, height, 
                        normalized_area, shape_difference], dtype=np.float32)
    
    def _calculate_features(self, class_num: float, x_center: float, y_center: float, 
                          width: float, height: float) -> np.ndarray:
        """
        Calculate all required features from bounding box data (for gated model).
        
        Args:
            class_num: Object class number
            x_center: Normalized x center coordinate
            y_center: Normalized y center coordinate  
            width: Normalized width
            height: Normalized height
            
        Returns:
            np.ndarray: Array of features [class, x_center, y_center, width, height, normalized_area, shape_difference]
        """
        # Calculate normalized area
        bounding_box_area = width * height
        normalized_area = bounding_box_area / self.image_area
        
        # Calculate shape difference (bounded between -1 and 1)
        epsilon = 1e-8  # Prevent division by zero
        shape_difference = (width - height) / (width + height + epsilon)
        
        return np.array([class_num, x_center, y_center, width, height, 
                        normalized_area, shape_difference], dtype=np.float32)
    
    def predict_distance(self, class_num: Union[int, float], x_center: float, 
                        y_center: float, width: float, height: float) -> Optional[float]:
        """
        Predict distance for a single detection using the appropriate model.
        
        Args:
            class_num: Object class number
            x_center: Normalized x center coordinate (0-1)
            y_center: Normalized y center coordinate (0-1)
            width: Normalized width (0-1)
            height: Normalized height (0-1)
            
        Returns:
            float: Predicted distance, or None if prediction fails
        """
        model = self._get_model_for_class(class_num)
        
        if model is None:
            model_type = "class 0" if class_num == 0 else "other classes"
            print(f"Error: Model for {model_type} not loaded.")
            return None
        
        try:
            # Calculate features based on class type
            if class_num == 0:
                # Class 0 model doesn't need class information
                features = self._calculate_features_class0(x_center, y_center, width, height)
            else:
                # Other classes model needs class information
                features = self._calculate_features(class_num, x_center, y_center, width, height)
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.tensor(features).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(input_tensor)
                distance = prediction.squeeze().cpu().item()
            
            return distance
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def predict_batch(self, detections: List[Tuple[Union[int, float], float, float, float, float]]) -> List[Optional[float]]:
        """
        Predict distances for multiple detections at once, using appropriate models.
        
        Args:
            detections: List of tuples (class_num, x_center, y_center, width, height)
            
        Returns:
            List[Optional[float]]: List of predicted distances (None for failed predictions)
        """
        if not detections:
            return []
        
        # Separate detections by class type
        class0_detections = []
        other_class_detections = []
        class0_indices = []
        other_class_indices = []
        
        for i, detection in enumerate(detections):
            class_num = detection[0]
            if class_num == 0:
                class0_detections.append(detection)
                class0_indices.append(i)
            else:
                other_class_detections.append(detection)
                other_class_indices.append(i)
        
        # Initialize results list
        results = [None] * len(detections)
        
        # Process class 0 detections
        if class0_detections and self.class0_loaded:
            class0_predictions = self._predict_batch_single_model(
                class0_detections, self.class0_model
            )
            for idx, pred in zip(class0_indices, class0_predictions):
                results[idx] = pred
        elif class0_detections and not self.class0_loaded:
            print("Warning: Class 0 detections found but class 0 model not loaded")
        
        # Process other class detections
        if other_class_detections and self.other_classes_loaded:
            other_predictions = self._predict_batch_single_model(
                other_class_detections, self.other_classes_model
            )
            for idx, pred in zip(other_class_indices, other_predictions):
                results[idx] = pred
        elif other_class_detections and not self.other_classes_loaded:
            print("Warning: Non-class-0 detections found but other classes model not loaded")
        
        return results
    
    def _predict_batch_single_model(self, detections: List[Tuple[Union[int, float], float, float, float, float]], 
                                   model: torch.nn.Module) -> List[Optional[float]]:
        """
        Predict distances for multiple detections using a single model.
        
        Args:
            detections: List of detections for this model
            model: The model to use for prediction
            
        Returns:
            List[Optional[float]]: List of predicted distances
        """
        try:
            # Calculate features for all detections
            features_list = []
            for class_num, x_center, y_center, width, height in detections:
                # Calculate features based on class type
                if class_num == 0:
                    # Class 0 model doesn't need class information
                    features = self._calculate_features_class0(x_center, y_center, width, height)
                else:
                    # Other classes model needs class information
                    features = self._calculate_features(class_num, x_center, y_center, width, height)
                
                features_list.append(features)
            
            # Stack into batch tensor
            batch_tensor = torch.tensor(np.stack(features_list)).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                predictions = model(batch_tensor)
                distances = predictions.squeeze().cpu().numpy()
            
            # Handle single prediction case
            if distances.ndim == 0:
                distances = [distances.item()]
            else:
                distances = distances.tolist()
            
            return distances
            
        except Exception as e:
            print(f"Error during batch prediction: {e}")
            return [None] * len(detections)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded models.
        
        Returns:
            dict: Model information
        """
        info = {
            'class0_loaded': self.class0_loaded,
            'other_classes_loaded': self.other_classes_loaded,
            'device': str(self.device),
            'class0_checkpoint_path': self.class0_checkpoint_path,
            'other_classes_checkpoint_path': self.other_classes_checkpoint_path,
            'class0_training_config': self.class0_training_config,
            'other_classes_training_config': self.other_classes_training_config
        }
        return info
    
    @property
    def is_loaded(self) -> bool:
        """Check if at least one model is loaded"""
        return self.class0_loaded or self.other_classes_loaded


# Convenience function for quick model loading
def load_distance_models(class0_checkpoint_path: str = 'distance_model_checkpoint_class0_only.pth',
                        other_classes_checkpoint_path: str = 'distance_model_checkpoint_no_class0.pth',
                        device: Optional[str] = None) -> Optional[CAVDualDistancePredictor]:
    """
    Quick function to load and return both distance prediction models.
    
    Args:
        class0_checkpoint_path: Path to class 0 model checkpoint
        other_classes_checkpoint_path: Path to other classes model checkpoint
        device: Device to use for inference
        
    Returns:
        CAVDualDistancePredictor: Loaded model instance, or None if loading failed
    """
    predictor = CAVDualDistancePredictor(class0_checkpoint_path, other_classes_checkpoint_path, device)
    if predictor.load_models():
        return predictor
    else:
        return None


# Backward compatibility - keep the old function name
def load_distance_model(checkpoint_path: str = 'distance_model_checkpoint_no_class0.pth', 
                       device: Optional[str] = None) -> Optional[CAVDualDistancePredictor]:
    """
    Backward compatibility function - loads dual models with default paths.
    
    Args:
        checkpoint_path: Path to other classes model checkpoint (for backward compatibility)
        device: Device to use for inference
        
    Returns:
        CAVDualDistancePredictor: Loaded model instance, or None if loading failed
    """
    return load_distance_models(
        class0_checkpoint_path='distance_model_checkpoint_class0_only.pth',
        other_classes_checkpoint_path=checkpoint_path,
        device=device
    )


# Example usage
if __name__ == "__main__":
    # Example of how to use the dual model system
    print("CAV Dual Distance Model - Example Usage")
    print("=" * 45)
    
    # Load both models
    predictor = load_distance_models()
    
    if predictor:
        # Example single predictions
        print("\nSingle Predictions:")
        print("-" * 20)
        
        # Test class 0 prediction
        distance_class0 = predictor.predict_distance(
            class_num=0,      # class 0
            x_center=0.5,     # center of image
            y_center=0.6,     # slightly below center
            width=0.3,        # 30% of image width
            height=0.4        # 40% of image height
        )
        print(f"Class 0 predicted distance: {distance_class0}")
        
        # Test other class prediction
        distance_other = predictor.predict_distance(
            class_num=1,      # e.g. speed limit sign
            x_center=0.5,     # center of image
            y_center=0.6,     # slightly below center
            width=0.3,        # 30% of image width
            height=0.4        # 40% of image height
        )
        print(f"Class 1 predicted distance: {distance_other}")
        
        # Example batch prediction with mixed classes
        print("\nBatch Predictions:")
        print("-" * 20)
        detections = [
            (0, 0.05738995, 0.6187029, 0.1026759, 0.03241112),     # class 0, expected 17.39995
            (14, 0.1794062, 0.4745811, 0.03907127, 0.03808923),  # class 14, expected 24.24867
            (0, 0.3076032, 0.951492, 0.1236521, 0.05581215),     # another class 0, expected 6.281461
            (2, 0.7214175, 0.4632267, 0.02344952, 0.03480754),  # 80 speed limit, expected 23.55716
            (3, 0.3481455, 0.4660687, 0.01601536, 0.02229566)   # 110 speed limit, expected 35.34863
        ]
        
        distances = predictor.predict_batch(detections)
        for i, (detection, distance) in enumerate(zip(detections, distances)):
            class_num = detection[0]
            print(f"Detection {i+1} (class {class_num}): {distance}")
        
        # Print model info
        print(f"\nModel Info:")
        print("-" * 20)
        info = predictor.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    else:
        print("Failed to load models!")