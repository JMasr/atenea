dependencies = ['torch']
import os
import torch


def silero_vad():
    """Silero Voice Activity Detector
    Returns the jit version of the model 
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    hub_dir = os.getcwd()
    torch.set_grad_enabled(False)
    model = torch.jit.load(f'{hub_dir}/../models/basic_vad/silero_vad.jit', map_location=torch.device('cpu'))
    model.eval()
    
    return model
