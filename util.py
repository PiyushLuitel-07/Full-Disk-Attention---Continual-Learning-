import time
import torch


def main():
    if not torch.cuda.is_available():
        print("No CUDA GPUs detected.")
        return

    num_gpus = torch.cuda.device_count()
    print("Available GPUs:")
    for i in range(num_gpus):
        free_b, total_b = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {free_b / (1024**3):.2f} GiB free / {total_b / (1024**3):.2f} GiB total")

    while True:
        gpu_input = input(f"\nSelect GPU (0-{num_gpus - 1}) [default 0]: ").strip()
        if gpu_input == "":
            gpu_id = 0
            break
        if gpu_input.isdigit() and 0 <= int(gpu_input) < num_gpus:
            gpu_id = int(gpu_input)
            break
        print(f"Invalid choice. Enter a number between 0 and {num_gpus - 1}.")

    free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
    mem_input = input(f"GiB to hold on GPU {gpu_id} (Enter for max minus 1 GiB): ").strip()
    if mem_input == "":
        alloc_bytes = max(0, free_bytes - 1024**3)
    else:
        alloc_bytes = int(float(mem_input) * (1024**3))

    try:
        buf = torch.empty(alloc_bytes, dtype=torch.uint8, device=f"cuda:{gpu_id}")
        print(
            f"\nHolding {buf.numel() / (1024**3):.2f} GiB ({buf.numel() / (1024**2):.0f} MiB) on GPU {gpu_id}."
        )
        print("Press Ctrl+C to release.")
        while True:
            time.sleep(3600)
    except torch.OutOfMemoryError:
        print(f"OOM: Not enough free memory on GPU {gpu_id} to allocate {alloc_bytes / (1024**3):.2f} GiB.")
    except KeyboardInterrupt:
        print("\nReleasing GPU memory and exiting.")


if __name__ == "__main__":
    main()