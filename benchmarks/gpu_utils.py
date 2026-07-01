import fcntl
import os
import subprocess

# Keep the lock handle alive for the process lifetime.
_gpu_lock_file = None


def select_gpu():
    """Select a CUDA GPU for the current process.

    GPUs are tried in descending free-memory order. The first available GPU lock
    is claimed and exposed as CUDA device 0. If all locks are already held, the
    process shares the GPU with the most free memory.

    Returns
    -------
        None. Sets ``CUDA_VISIBLE_DEVICES`` when a GPU can be selected.
    """
    global _gpu_lock_file

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            idx, mem_free = line.split(",")
            gpus.append((int(idx.strip()), int(mem_free.strip())))
        gpus.sort(key=lambda x: x[1], reverse=True)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return

    lock_dir = "/tmp/snakemake_gpu_locks"
    os.makedirs(lock_dir, exist_ok=True)

    for gpu_idx, mem_free in gpus:
        lock_path = os.path.join(lock_dir, f"gpu_{gpu_idx}.lock")
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _gpu_lock_file = lock_file
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            _configure_tf(gpu_idx)
            print(f"[gpu_utils] Assigned GPU {gpu_idx} ({mem_free} MiB free)")
            return
        except OSError:
            lock_file.close()

    if gpus:
        gpu_idx, mem_free = gpus[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        _configure_tf(gpu_idx)
        print(f"[gpu_utils] All GPU locks are held; sharing GPU {gpu_idx} ({mem_free} MiB free)")
        return
    print("[gpu_utils] All GPUs locked by other jobs; falling back to default device selection.")


def _configure_tf(gpu_idx):
    """Restrict TensorFlow to a selected GPU if TensorFlow is loaded.

    Args:
        gpu_idx: Physical GPU index selected by ``select_gpu``.

    Returns
    -------
        None.
    """
    try:
        import tensorflow as tf

        physical_gpus = tf.config.list_physical_devices("GPU")
        if physical_gpus and gpu_idx < len(physical_gpus):
            tf.config.set_visible_devices(physical_gpus[gpu_idx], "GPU")
    except (ImportError, RuntimeError):
        pass
