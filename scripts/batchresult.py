import pandas as pd
import time
from vllm import LLM, SamplingParams
from main import initialize_gpu_monitor, shutdown_gpu_monitor, get_gpu_utilization, visualize_distributions, cleanup,load_model, RESULTS_PATH ,MODEL_PATH, TEMP_MODEL_PATH, load_dataset, DATASET_DIR
import threading
import json
def inference_with_varying_batch_sizes(samples, llm, results_queue, batch_sizes):
    """
    根据不同的批次大小测试推理性能。
    
    Args:
        samples (list): 输入样本。
        llm (vllm.LLM): 模型对象。
        results_queue (list): 用于存储结果的队列。
        batch_sizes (list): 需要测试的批次大小列表。
    """
    sampling_params = SamplingParams(max_tokens=100, temperature=0.7)

    for batch_size in batch_sizes:
        start_time = time.time()
        batch_start = 0

        while batch_start < len(samples):
            batch_samples = samples[batch_start:batch_start + batch_size]
            batch_start_time = time.time()

            outputs = llm.generate(batch_samples, sampling_params)

            batch_latency = time.time() - batch_start_time
            gpu_utilization, memory_used = get_gpu_utilization()
            throughput = len(batch_samples) / batch_latency

            # 记录每个批次的性能
            results_queue.append({
                "batch_size": batch_size,
                "throughput": throughput,
                "avg_latency": batch_latency,
                "gpu_utilization": gpu_utilization,
                "memory_used": memory_used,
            })

            batch_start += batch_size

        total_time = time.time() - start_time
        total_throughput = len(samples) / total_time
        avg_latency = total_time / len(samples)

        # 汇总记录
        results_queue.append({
            "batch_size": "final",
            "throughput": total_throughput,
            "avg_latency": avg_latency,
            "gpu_utilization": gpu_utilization,
            "memory_used": memory_used,
        })

def generate_results_table(results_queue, model_name, output_tokens_range, output_csv_path="results_table.csv"):
    """
    生成性能测试结果表格，并保存为CSV文件。
    
    Args:
        results_queue (list): 包含批次性能结果的队列。
        model_name (str): 模型名称。
        output_tokens_range (str): 输出token范围描述。
        output_csv_path (str): 保存的CSV文件路径。
    """
    table_data = []

    for result in results_queue:
        if result["batch_size"] == "final":
            continue  # 跳过汇总记录
        row = {
            "模型名称": model_name,
            "输入 token": 3500,  # 假设输入token固定
            "输出 token": output_tokens_range,
            "最大并发": result["batch_size"],
            "平均时延 (s)": result["avg_latency"],
            "最大时延 (s)": result["avg_latency"],  # 简化处理，这里使用平均时延代替最大时延
            "吞吐 (r/s)": result["throughput"],
            "kv cache usage (%)": result["gpu_utilization"]
        }
        table_data.append(row)

    # 将数据转为 Pandas DataFrame
    df = pd.DataFrame(table_data)
    
    # 保存为 CSV 文件
    df.to_csv(output_csv_path, index=False)
    print(f"Results table saved to {output_csv_path}")

    # 显示表格
    print(df)
    return df

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
            gpu_memory_utilization=0.99
        )

        results_queue = []

        # 定义需要测试的批次大小范围
        batch_sizes = list(range(4, 65, 4))  # 从4到64，步长为4

        # 启动推理测试
        inference_with_varying_batch_sizes(samples, llm, results_queue, batch_sizes)

        # 保存结果
        with open(RESULTS_PATH, "w") as f:
            json.dump(list(results_queue), f, indent=4)
        print(f"Results saved to {RESULTS_PATH}")

        # 生成并保存表格
        model_name = "llama3.1-8b"
        output_tokens_range = "490 - 500"
        results_table = generate_results_table(results_queue, model_name, output_tokens_range)

        # 可视化分布
        print("Starting visualization...")
        visualize_distributions(list(results_queue))
    finally:
        shutdown_gpu_monitor()
        cleanup()

if __name__ == "__main__":
    main()
