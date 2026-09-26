import torch
import numpy as np

class MockModel:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def __init__(self, model_type="flux"):
        self.model_type = model_type
        self.model_sampling = MockModelSampling()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def get_model_object(self, name):
        return self

    def __getattr__(self, name):
        if name == "model":
            return self
        return None

class MockModelSampling:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def __init__(self):
        self.shift = lambda x: x * 1.15 # Dummy shift

    def sigmas(self, steps):
        return torch.linspace(1, 0, steps)

class MockVAE:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def decode(self, latent):
        return torch.zeros((latent.shape[0], latent.shape[2]*8, latent.shape[3]*8, 3))

class MockConditioning:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def __init__(self):
        pass

def mock_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    return latent_image # No-op sampler

# Simulating comfy.model_management
class MockModelManagement:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    @staticmethod
    def load_model_gpu(model):
        pass
    
    @staticmethod
    def soft_empty_cache():
        pass
        
    @staticmethod
    def unload_all_models():
        pass

    @staticmethod
    def get_torch_device():
        return torch.device("cpu")

# Simulating comfy.sample
class MockSampleModule:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    @staticmethod
    def sample(*args, **kwargs):
        return mock_sample(*args, **kwargs)

# Simulating node registry for testing
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
