import os

UUID_DICT = {
    # Get by running nvidia-smi -L on each machine
    # Needed for SLURM mappings
    # Based on docs: https://github.com/johnrso/spawnnet/blob/main/RLAfford/README.md#slurm-gpu-mapping
    "gr007.hpc.nyu.edu":
        {
            "GPU-3ac267aa-d7e9-15e7-4a43-b75d67072417":
                "0",
        },
    "gr014.hpc.nyu.edu":
        {
            "GPU-1c6258c7-496e-f34a-fe02-6f553df6b74f":
                "0",
        },
    "gr018.hpc.nyu.edu":
        {
            "GPU-9f690a1b-1ee3-9f59-cd54-60706a7a7fe8":
                "0",
        },
    "gr007.hpc.nyu.edu":
        {
            "GPU-7e2db917-f097-025b-fd85-44cd25fb8acc":
                "0",
        },
    "gr019.hpc.nyu.edu":
        {
            "GPU-5cd5215b-bff4-11e3-9280-68e10e72ef32":
                "0",
        },
    "gr052.hpc.nyu.edu":
        {
            "GPU-c9d19c79-1247-15a8-e890-a3c2c1ab9e11":
                "0",
        },
    "gr017.hpc.nyu.edu":
        {
            "GPU-e217f9ac-5363-d301-c065-8dc565a3f43b":
                "0",
        },
}

def get_hostname():
    import socket
    hostname = socket.gethostname()
    return hostname

def get_gpu_uuid():
    import os
    gpu_info_list = os.popen('nvidia-smi -L').read().split('\n')
    uuids = []
    for gpu_info in gpu_info_list:
        if 'UUID' in gpu_info:
            gpu_uuid = gpu_info.split('UUID:')[1].strip()
            gpu_uuid = gpu_uuid.rstrip(')')
            uuids.append(gpu_uuid)
    return uuids

def get_graphics_gpu_ids():
    uuids = get_gpu_uuid()
    # If CUDA_VISIBLE_DEVICES is set, use it
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        uuids = [uuids[i] for i in gpu_ids]

    hostname = get_hostname()
    # import pdb; pdb.set_trace()
    gpu_ids = [UUID_DICT[hostname][uuid] for uuid in uuids]
    return gpu_ids

def vulkan_cuda_idxes(mode, num_gpus):
    if mode == 'slurm':
        from gpu_info import get_graphics_gpu_ids
        vulkan_gpu_idxes = get_graphics_gpu_ids()[:num_gpus]
        cuda_gpu_idxes = list(range(num_gpus))
        print(f"CUDA GPUs: {cuda_gpu_idxes}")
        print(f"Vulkan GPUs: {vulkan_gpu_idxes}")
    elif mode == "basic":
        vulkan_gpu_idxes = [int(x) for x in os.environ["WHICH_GPUS"].split(",")][:num_gpus]
        cuda_gpu_idxes = vulkan_gpu_idxes
    else:
        raise NotImplementedError(f"Unsupported launcher: {os.environ['HYDRA_LAUNCHER']}")
    
    return cuda_gpu_idxes, vulkan_gpu_idxes

if __name__ == '__main__':
    g_ids = get_graphics_gpu_ids()
    print(g_ids)
    
