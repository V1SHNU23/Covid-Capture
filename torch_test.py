import torch  

# Check if CUDA (GPU) is available
print(torch.cuda.is_available())

# Get the number of available GPUs
print(torch.cuda.device_count())

# Get the current device
print(torch.cuda.current_device())

# Get the device by index (0 in this case)
print(torch.cuda.device(0))

# Get the name of the device (GPU)
print(torch.cuda.get_device_name(0))
