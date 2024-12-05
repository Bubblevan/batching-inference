from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# def real_time_monitor(results_queue, interval=1000):
#     """
#     实时监控推理过程的吞吐量、延迟和GPU利用率。

#     Args:
#         results_queue (list): 保存每批次结果的队列。
#         interval (int): 图表刷新间隔（毫秒）。
#     """
#     fig, axes = plt.subplots(3, 1, figsize=(10, 15))
#     plt.subplots_adjust(hspace=0.5)

#     # 初始化图表
#     throughput_ax = axes[0]
#     latency_ax = axes[1]
#     gpu_util_ax = axes[2]

#     throughput_ax.set_title("Throughput Over Time")
#     throughput_ax.set_xlabel("Batch")
#     throughput_ax.set_ylabel("Throughput (req/s)")
#     latency_ax.set_title("Latency Over Time")
#     latency_ax.set_xlabel("Batch")
#     latency_ax.set_ylabel("Latency (s)")
#     gpu_util_ax.set_title("GPU Utilization Over Time")
#     gpu_util_ax.set_xlabel("Batch")
#     gpu_util_ax.set_ylabel("GPU Utilization (%)")

#     throughput_line, = throughput_ax.plot([], [], label="Throughput", color="blue")
#     latency_line, = latency_ax.plot([], [], label="Latency", color="orange")
#     gpu_util_line, = gpu_util_ax.plot([], [], label="GPU Utilization", color="green")

#     throughput_ax.legend()
#     latency_ax.legend()
#     gpu_util_ax.legend()

#     # 动态更新数据
#     def update(frame):
#         if not results_queue:
#             return throughput_line, latency_line, gpu_util_line

#         # 获取实时数据
#         throughputs = [result["throughput"] for result in results_queue]
#         latencies = [result["avg_latency"] for result in results_queue]
#         gpu_utils = [result["gpu_utilization"] for result in results_queue]

#         x_data = list(range(len(throughputs)))

#         # 更新曲线数据
#         throughput_line.set_data(x_data, throughputs)
#         latency_line.set_data(x_data, latencies)
#         gpu_util_line.set_data(x_data, gpu_utils)

#         # 调整坐标轴
#         throughput_ax.set_xlim(0, max(x_data) + 1)
#         throughput_ax.set_ylim(0, max(throughputs) * 1.1)
#         latency_ax.set_xlim(0, max(x_data) + 1)
#         latency_ax.set_ylim(0, max(latencies) * 1.1)
#         gpu_util_ax.set_xlim(0, max(x_data) + 1)
#         gpu_util_ax.set_ylim(0, 100)

#         return throughput_line, latency_line, gpu_util_line

#     ani = FuncAnimation(fig, update, interval=interval)
#     plt.show()

def visualize_distributions(results):
    """
    使用箱线图和直方图展示延迟和吞吐量的分布。

    Args:
        results (list): 推理结果列表。
    """
    throughputs = [result["throughput"] for result in results]
    latencies = [result["avg_latency"] for result in results]

    # 吞吐量分布
    plt.figure(figsize=(10, 6))
    plt.boxplot(throughputs, vert=False, patch_artist=True)
    plt.title("Throughput Distribution")
    plt.xlabel("Throughput (req/s)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(throughputs, bins=10, color="skyblue", edgecolor="black")
    plt.title("Throughput Histogram")
    plt.xlabel("Throughput (req/s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # 延迟分布
    plt.figure(figsize=(10, 6))
    plt.boxplot(latencies, vert=False, patch_artist=True)
    plt.title("Latency Distribution")
    plt.xlabel("Latency (s)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=10, color="orange", edgecolor="black")
    plt.title("Latency Histogram")
    plt.xlabel("Latency (s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
