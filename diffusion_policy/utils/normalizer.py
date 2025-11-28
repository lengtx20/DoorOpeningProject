"""
Data normalization utilities for diffusion policy.
"""

import torch
import numpy as np
from typing import Dict, Union


class LinearNormalizer:
    """
    Linear normalizer that scales data to [-1, 1] or [0, 1] range.
    """

    def __init__(self, mode: str = 'limits'):
        """
        Args:
            mode: 'limits' for min-max normalization, 'gaussian' for z-score
        """
        self.mode = mode
        self.params = {}

    def fit(self, data: Union[np.ndarray, torch.Tensor], key: str = 'default'):
        """
        Fit normalizer to data.

        Args:
            data: (N, D) array of data
            key: Identifier for this data type (e.g., 'obs', 'action')
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if self.mode == 'limits':
            # Min-max normalization to [-1, 1]
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)

            # Avoid division by zero
            scale = max_val - min_val
            scale[scale < 1e-8] = 1.0

            self.params[key] = {
                'min': min_val,
                'max': max_val,
                'scale': scale,
            }

        elif self.mode == 'gaussian':
            # Z-score normalization
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            # Avoid division by zero
            std[std < 1e-8] = 1.0

            self.params[key] = {
                'mean': mean,
                'std': std,
            }

    def normalize(self, data: Union[np.ndarray, torch.Tensor],
                  key: str = 'default') -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize data using fitted parameters.

        Args:
            data: Data to normalize
            key: Which normalization parameters to use
        Returns:
            Normalized data
        """
        is_torch = isinstance(data, torch.Tensor)
        device = data.device if is_torch else None

        if is_torch:
            data_np = data.cpu().numpy()
        else:
            data_np = data

        if key not in self.params:
            raise ValueError(f"Normalizer not fitted for key '{key}'")

        if self.mode == 'limits':
            params = self.params[key]
            # Scale to [0, 1] then to [-1, 1]
            normalized = (data_np - params['min']) / params['scale']
            normalized = normalized * 2 - 1

        elif self.mode == 'gaussian':
            params = self.params[key]
            normalized = (data_np - params['mean']) / params['std']

        if is_torch:
            return torch.from_numpy(normalized).float().to(device)
        return normalized

    def denormalize(self, data: Union[np.ndarray, torch.Tensor],
                    key: str = 'default') -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize data back to original scale.

        Args:
            data: Normalized data
            key: Which normalization parameters to use
        Returns:
            Denormalized data
        """
        is_torch = isinstance(data, torch.Tensor)
        device = data.device if is_torch else None

        if is_torch:
            data_np = data.cpu().numpy()
        else:
            data_np = data

        if key not in self.params:
            raise ValueError(f"Normalizer not fitted for key '{key}'")

        if self.mode == 'limits':
            params = self.params[key]
            # Scale from [-1, 1] to [0, 1] then to original range
            denormalized = (data_np + 1) / 2
            denormalized = denormalized * params['scale'] + params['min']

        elif self.mode == 'gaussian':
            params = self.params[key]
            denormalized = data_np * params['std'] + params['mean']

        if is_torch:
            return torch.from_numpy(denormalized).float().to(device)
        return denormalized

    def save(self, path: str):
        """Save normalizer parameters to file."""
        np.savez(path, **{k: v for k, params in self.params.items()
                          for k, v in params.items()})

    def load(self, path: str):
        """Load normalizer parameters from file."""
        data = np.load(path)
        # Reconstruct params dict
        # This is simplified; may need refinement based on actual save format
        for key in ['default', 'obs', 'action']:
            if self.mode == 'limits':
                if f'{key}_min' in data:
                    self.params[key] = {
                        'min': data[f'{key}_min'],
                        'max': data[f'{key}_max'],
                        'scale': data[f'{key}_scale'],
                    }
            elif self.mode == 'gaussian':
                if f'{key}_mean' in data:
                    self.params[key] = {
                        'mean': data[f'{key}_mean'],
                        'std': data[f'{key}_std'],
                    }
