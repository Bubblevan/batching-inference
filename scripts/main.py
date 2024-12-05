# main.py
import os
import sys
import time
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from vllm import LLM, SamplingParams
import glob
import torch.distributed as dist
import threading
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pynvml

def initialize_gpu_monitor():
    """初始化 GPU 监控"""
    pynvml.nvmlInit()

def shutdown_gpu_monitor():
    """关闭 GPU 监控"""
    pynvml.nvmlShutdown()

def get_gpu_utilization():
    """
    获取当前 GPU 利用率和显存使用量。

    Returns:
        utilization (int): GPU 利用率（%）
        memory_used (int): 显存使用量（MB）
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用第一个 GPU
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    memory_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used // 1024 // 1024  # 转换为 MB
    return utilization, memory_used

def real_time_monitor(results_queue, interval=20000):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    plt.subplots_adjust(hspace=0.5)

    throughput_ax = axes[0]
    latency_ax = axes[1]
    gpu_util_ax = axes[2]

    throughput_ax.set_title("Throughput Over Time")
    throughput_ax.set_xlabel("Batch")
    throughput_ax.set_ylabel("Throughput (req/s)")
    latency_ax.set_title("Latency Over Time")
    latency_ax.set_xlabel("Batch")
    latency_ax.set_ylabel("Latency (s)")
    gpu_util_ax.set_title("GPU Utilization Over Time")
    gpu_util_ax.set_xlabel("Batch")
    gpu_util_ax.set_ylabel("GPU Utilization (%)")

    throughput_line, = throughput_ax.plot([], [], label="Throughput", color="blue")
    latency_line, = latency_ax.plot([], [], label="Latency", color="orange")
    gpu_util_line, = gpu_util_ax.plot([], [], label="GPU Utilization", color="green")

    throughput_ax.legend()
    latency_ax.legend()
    gpu_util_ax.legend()

    def update(frame):
        if not results_queue:
            return throughput_line, latency_line, gpu_util_line

        throughputs = [result["throughput"] for result in results_queue]
        latencies = [result["avg_latency"] for result in results_queue]
        gpu_utils = [result["gpu_utilization"] for result in results_queue]

        x_data = list(range(len(throughputs)))

        throughput_line.set_data(x_data, throughputs)
        latency_line.set_data(x_data, latencies)
        gpu_util_line.set_data(x_data, gpu_utils)

        throughput_ax.set_xlim(0, max(x_data) + 1)
        throughput_ax.set_ylim(0, max(throughputs) * 1.1)
        latency_ax.set_xlim(0, max(x_data) + 1)
        latency_ax.set_ylim(0, max(latencies) * 1.1)
        gpu_util_ax.set_xlim(0, max(x_data) + 1)
        gpu_util_ax.set_ylim(0, 100)

        return throughput_line, latency_line, gpu_util_line

    ani = FuncAnimation(fig, update, interval=interval)
    ani.save("real_time_monitor_animation.mp4", writer="ffmpeg")  # 可选：保存为视频

def visualize_distributions(results):
    if not results:
        print("No results to visualize.")
        return

    throughputs = [result["throughput"] for result in results if "throughput" in result]
    latencies = [result["avg_latency"] for result in results if "avg_latency" in result]

    if not throughputs or not latencies:
        print("Insufficient data for visualization.")
        return

    output_dir = "./output_plots"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.boxplot(throughputs, vert=False, patch_artist=True)
    plt.title("Throughput Distribution")
    plt.xlabel("Throughput (req/s)")
    plt.tight_layout()
    throughput_path = os.path.join(output_dir, "throughput_distribution.png")
    plt.savefig(throughput_path)
    print(f"Throughput distribution saved to {throughput_path}")

    plt.figure(figsize=(10, 6))
    plt.hist(throughputs, bins=10, color="skyblue", edgecolor="black")
    plt.title("Throughput Histogram")
    plt.xlabel("Throughput (req/s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    throughput_hist_path = os.path.join(output_dir, "throughput_histogram.png")
    plt.savefig(throughput_hist_path)
    print(f"Throughput histogram saved to {throughput_hist_path}")

    plt.figure(figsize=(10, 6))
    plt.boxplot(latencies, vert=False, patch_artist=True)
    plt.title("Latency Distribution")
    plt.xlabel("Latency (s)")
    plt.tight_layout()
    latency_path = os.path.join(output_dir, "latency_distribution.png")
    plt.savefig(latency_path)
    print(f"Latency distribution saved to {latency_path}")

    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=10, color="orange", edgecolor="black")
    plt.title("Latency Histogram")
    plt.xlabel("Latency (s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    latency_hist_path = os.path.join(output_dir, "latency_histogram.png")
    plt.savefig(latency_hist_path)
    print(f"Latency histogram saved to {latency_hist_path}")

vllm_path = os.path.abspath('../vllm')
sys.path.append(vllm_path)

DATASET_DIR = "/root/autodl-pub/datasets/lmsys-chat-1m/data"
MODEL_PATH = "/root/autodl-pub/Meta-Llama-3.1-8B-Instruct"
TEMP_MODEL_PATH = "/root/project/temp_model"
RESULTS_PATH = "/root/project/scripts/results.json"

def load_model(model_path):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = 'left'

    return tokenizer, model, device

def load_dataset(dataset_dir, max_samples=1000):
    print("Loading dataset...")
    samples = []
    parquet_files = glob.glob(os.path.join(dataset_dir, "*.parquet"))

    if not parquet_files:
        print(f"No .parquet files found in {dataset_dir}. Exiting.")
        return samples

    for file_path in parquet_files:
        table = pq.read_table(file_path)
        df = table.to_pandas()

        for _, row in df.iterrows():
            conversation = row.get("conversation")
            if isinstance(conversation, np.ndarray):
                conversation = conversation.tolist()

            if isinstance(conversation, list):
                for turn in conversation:
                    if turn.get("role") == "user":
                        content = turn.get("content", "").strip()
                        if content:
                            samples.append(content)
                        if len(samples) >= max_samples:
                            break
            if len(samples) >= max_samples:
                break

        if len(samples) >= max_samples:
            break

    print(f"Loaded {len(samples)} samples.")
    return samples

def group_by_length(samples, tokenizer, thresholds=[32, 128, 512]):
    lengths = [len(tokenizer.encode(sample)) for sample in samples]
    groups = {t: [] for t in thresholds + ["others"]}
    for sample, length in zip(samples, lengths):
        for threshold in thresholds:
            if length <= threshold:
                groups[threshold].append(sample)
                break
        else:
            groups["others"].append(sample)
    return groups

def inference_with_vllm_continuous_batching(samples, llm, results_queue):
    """
    Perform inference using vLLM's built-in continuous batching for optimized GPU utilization.
    
    Args:
        samples (list): List of input samples.
        llm (vllm.LLM): vLLM model object.
        results_queue (list): List to store the results.
    """
    sampling_params = SamplingParams(max_tokens=100, temperature=0.7)

    # Start processing samples with vLLM
    start_time = time.time()
    for batch_start in range(0, len(samples), 64):  # Batch size dynamically adjusted in vLLM
        batch_samples = samples[batch_start:batch_start + 64]
        batch_start_time = time.time()

        # Perform inference with vLLM
        outputs = llm.generate(batch_samples, sampling_params)

        batch_latency = time.time() - batch_start_time
        gpu_utilization, memory_used = get_gpu_utilization()
        throughput = len(batch_samples) / batch_latency

        # Collect results
        results_queue.append({
            "batch_size": len(batch_samples),
            "throughput": throughput,
            "avg_latency": batch_latency,
            "gpu_utilization": gpu_utilization,
            "memory_used": memory_used
        })

    total_time = time.time() - start_time
    total_throughput = len(samples) / total_time
    avg_latency = total_time / len(samples)

    results_queue.append({
        "batch_size": "final",
        "throughput": total_throughput,
        "avg_latency": avg_latency,
        "gpu_utilization": gpu_utilization,
        "memory_used": memory_used
    })

def cleanup():
    """销毁分布式进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    try:
        initialize_gpu_monitor()

        tokenizer, model, device = load_model(MODEL_PATH)

        model.save_pretrained(TEMP_MODEL_PATH)
        tokenizer.save_pretrained(TEMP_MODEL_PATH)

        samples = load_dataset(DATASET_DIR)
        print(f"First 5 samples: {samples[:5]}")
        if not samples:
            print("No samples loaded. Exiting.")
            return

        llm = LLM(
            model=TEMP_MODEL_PATH,
            tokenizer=TEMP_MODEL_PATH,
            device=device,
            max_model_len=100000,  
            gpu_memory_utilization=0.95  
        )

        results_queue = []

        inference_thread = threading.Thread(
            target=inference_with_vllm_continuous_batching,
            args=(samples, llm, results_queue)
        )

        inference_thread.start()

        inference_thread.join()

        with open(RESULTS_PATH, "w") as f:
            json.dump(list(results_queue), f, indent=4)
        print(f"Results saved to {RESULTS_PATH}")
        print("Starting visualization...")
        visualize_distributions(list(results_queue))
    finally:
        shutdown_gpu_monitor()
        cleanup()

if __name__ == "__main__":
    main()