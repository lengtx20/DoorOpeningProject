import torch
import numpy as np


class Normalizer:
    def __init__(self, stats, eps=1e-6):
        self.stats = stats
        self.eps = eps

    def _linear(self, tensor, key, device):
        data_min = self.stats[key]['min'].to(device)
        data_max = self.stats[key]['max'].to(device)
        scale = torch.clamp(data_max - data_min, min=self.eps)
        return 2.0 * (tensor - data_min) / scale - 1.0

    def _inv_linear(self, tensor, key, device):
        data_min = self.stats[key]['min'].to(device)
        data_max = self.stats[key]['max'].to(device)
        scale = torch.clamp(data_max - data_min, min=self.eps)
        return 0.5 * (tensor + 1.0) * scale + data_min

    def normalize(self, batch):
        device = batch['action'].device
        batch['action'] = self._linear(batch['action'], 'action', device)

        if 'obs' in batch:
            batch['obs'] = self._linear(batch['obs'], 'obs', device)

        if 'agent_pos' in self.stats and 'agent_pos' in batch:
            if batch['agent_pos'].nelement() > 0:
                batch['agent_pos'] = self._linear(batch['agent_pos'], 'agent_pos', device)

        return batch

    def unnormalize_action(self, action):
        device = action.device
        return self._inv_linear(action, 'action', device)


class LinearNormalizer:
    def __init__(self, mode='limits', eps=1e-6):
        assert mode == 'limits', "Only 'limits' mode is supported."
        self.mode = mode
        self.eps = eps
        self.params = {}

    def normalize(self, tensor, key):
        params = self.params[key]
        data_min = params['min']
        scale = np.maximum(params['scale'], self.eps)
        if isinstance(tensor, torch.Tensor):
            data_min_t = torch.as_tensor(data_min, dtype=tensor.dtype, device=tensor.device)
            scale_t = torch.as_tensor(scale, dtype=tensor.dtype, device=tensor.device)
            scale_t = torch.clamp(scale_t, min=self.eps)
            return 2.0 * (tensor - data_min_t) / scale_t - 1.0
        else:
            return 2.0 * (tensor - data_min) / scale - 1.0

    def denormalize(self, tensor, key):
        params = self.params[key]
        data_min = params['min']
        scale = np.maximum(params['scale'], self.eps)
        if isinstance(tensor, torch.Tensor):
            data_min_t = torch.as_tensor(data_min, dtype=tensor.dtype, device=tensor.device)
            scale_t = torch.as_tensor(scale, dtype=tensor.dtype, device=tensor.device)
            scale_t = torch.clamp(scale_t, min=self.eps)
            return 0.5 * (tensor + 1.0) * scale_t + data_min_t
        else:
            return 0.5 * (tensor + 1.0) * scale + data_min

