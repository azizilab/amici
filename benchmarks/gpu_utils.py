import fcntl
import os
import subprocess

# Module-level reference keeps the lock file open (and the OS-level lock held)
# until the process exits, even if the caller doesn't keep a reference.
_gpu_lock_file = None


def select_gpu():
    """Find a free GPU, claim it with a file lock, and set CUDA_VISIBLE_DEVICES.

    Queries nvidia-smi for free memory, tries each GPU (most-free first), and
    acquires an exclusive non-blocking flock on /tmp/snakemake_gpu_locks/gpu_N.lock.
    The lock is held for the lifetime of the process so that parallel Snakemake
    jobs cannot claim the same device.

    Sets os.environ["CUDA_VISIBLE_DEVICES"] so that PyTorch, TensorFlow, and
    scvi-tools all see only the chosen GPU as device 0.

    Falls back silently if nvidia-smi is unavailable (CPU-only environment).
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
        gpus.sort(key=lambda x: x[1], reverse=True)  # prefer GPU with most free memory
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return  # no nvidia-smi; leave device selection to defaults

    lock_dir = "/tmp/snakemake_gpu_locks"
    os.makedirs(lock_dir, exist_ok=True)

    for gpu_idx, mem_free in gpus:
        lock_path = os.path.join(lock_dir, f"gpu_{gpu_idx}.lock")
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Lock acquired — keep file open so the OS holds it until process exits.
            _gpu_lock_file = lock_file
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            _configure_tf(gpu_idx)
            print(f"[gpu_utils] Assigned GPU {gpu_idx} ({mem_free} MiB free)")
            return
        except OSError:
            lock_file.close()

    # All GPUs are locked by sibling jobs. Fall back to default (likely GPU 0).
    print("[gpu_utils] All GPUs locked by other jobs; falling back to default device selection.")


def _configure_tf(gpu_idx):
    """Restrict TensorFlow to the selected GPU if TF is already imported."""
    try:
        import tensorflow as tf

        physical_gpus = tf.config.list_physical_devices("GPU")
        if physical_gpus and gpu_idx < len(physical_gpus):
            tf.config.set_visible_devices(physical_gpus[gpu_idx], "GPU")
    except (ImportError, RuntimeError):
        pass
