import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better-looking plots
plt.style.use("default")
sns.set_palette("husl")

# Data extracted from benchmark results
tokens = np.array([100, 200, 300, 400, 500, 600])

# Qwen3-Torch performance data
torch_latency = np.array([3.759, 11.289, 25.540, 55.232, 257.532, 440.425])
torch_tokens_per_sec = np.array([29.27, 19.50, 12.69, 7.83, 2.11, 1.45])

# FastQwen3-CUDA performance data
cuda_latency = np.array([1.878, 5.708, 11.419, 20.784, 29.097, 47.594])
cuda_tokens_per_sec = np.array([58.04, 39.25, 28.74, 21.20, 18.73, 13.62])

# Calculate speedups
speedup_latency = torch_latency / cuda_latency
speedup_throughput = cuda_tokens_per_sec / torch_tokens_per_sec

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "FastQwen3 CUDA Kernel Performance Analysis", fontsize=16, fontweight="bold"
)

# Plot 1: Latency Comparison
ax1.plot(
    tokens,
    torch_latency,
    "o-",
    linewidth=2,
    markersize=8,
    label="Qwen3-Torch",
    color="#e74c3c",
)
ax1.plot(
    tokens,
    cuda_latency,
    "s-",
    linewidth=2,
    markersize=8,
    label="FastQwen3-CUDA",
    color="#2ecc71",
)
ax1.set_xlabel("Number of Tokens")
ax1.set_ylabel("Latency (seconds)")
ax1.set_title("Latency Comparison", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale("log")

# Plot 2: Throughput Comparison
ax2.plot(
    tokens,
    torch_tokens_per_sec,
    "o-",
    linewidth=2,
    markersize=8,
    label="Qwen3-Torch",
    color="#e74c3c",
)
ax2.plot(
    tokens,
    cuda_tokens_per_sec,
    "s-",
    linewidth=2,
    markersize=8,
    label="FastQwen3-CUDA",
    color="#2ecc71",
)
ax2.set_xlabel("Number of Tokens")
ax2.set_ylabel("Throughput (tokens/second)")
ax2.set_title("Throughput Comparison", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Speedup Analysis
bars = ax3.bar(
    tokens, speedup_latency, color="#3498db", alpha=0.7, edgecolor="black", linewidth=1
)
ax3.set_xlabel("Number of Tokens")
ax3.set_ylabel("Speedup Factor (x)")
ax3.set_title("Latency Speedup (Higher is Better)", fontweight="bold")
ax3.grid(True, alpha=0.3)

# Add value labels on bars
for bar, speedup in zip(bars, speedup_latency):
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{speedup:.2f}x",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Plot 4: Throughput Improvement
bars2 = ax4.bar(
    tokens,
    speedup_throughput,
    color="#9b59b6",
    alpha=0.7,
    edgecolor="black",
    linewidth=1,
)
ax4.set_xlabel("Number of Tokens")
ax4.set_ylabel("Throughput Improvement (x)")
ax4.set_title("Throughput Improvement (Higher is Better)", fontweight="bold")
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, speedup in zip(bars2, speedup_throughput):
    height = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{speedup:.2f}x",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.tight_layout()
plt.show()

# Create a summary table
print("📊 FastQwen3 CUDA Performance Summary")
print("=" * 60)
print(
    f"{'Tokens':<8} {'Torch (s)':<10} {'CUDA (s)':<10} {'Speedup':<10} {'Efficiency'}"
)
print("-" * 60)
for i in range(len(tokens)):
    efficiency = (speedup_latency[i] - 1) * 100  # Percentage improvement
    print(
        f"{tokens[i]:<8} {torch_latency[i]:<10.2f} {cuda_latency[i]:<10.2f} {speedup_latency[i]:<10.2f} {efficiency:.1f}%"
    )

print("\n🚀 Key Insights:")
print(
    f"• Maximum speedup achieved: {max(speedup_latency):.2f}x at {tokens[np.argmax(speedup_latency)]} tokens"
)
print(f"• Average speedup across all tests: {np.mean(speedup_latency):.2f}x")
print(f"• Performance scales better with longer sequences")
print(
    f"• At 600 tokens: {speedup_latency[-1]:.2f}x faster, saving {(torch_latency[-1] - cuda_latency[-1])/60:.1f} minutes per inference"
)

# Additional analysis plot - Performance scaling
fig2, ax5 = plt.subplots(1, 1, figsize=(12, 8))

# Plot both latency curves with filled area between them
ax5.fill_between(
    tokens,
    torch_latency,
    cuda_latency,
    alpha=0.3,
    color="green",
    label="Performance Gain",
)
ax5.plot(
    tokens,
    torch_latency,
    "o-",
    linewidth=3,
    markersize=10,
    label="Qwen3-Torch",
    color="#e74c3c",
)
ax5.plot(
    tokens,
    cuda_latency,
    "s-",
    linewidth=3,
    markersize=10,
    label="FastQwen3-CUDA",
    color="#2ecc71",
)

ax5.set_xlabel("Number of Tokens", fontsize=14)
ax5.set_ylabel("Latency (seconds)", fontsize=14)
ax5.set_title(
    "FastQwen3 CUDA: Dramatic Performance Improvement", fontsize=16, fontweight="bold"
)
ax5.legend(fontsize=12)
ax5.grid(True, alpha=0.3)
ax5.set_yscale("log")

# Add annotations for key points
ax5.annotate(
    f"{speedup_latency[-1]:.1f}x faster\nat 600 tokens",
    xy=(tokens[-1], cuda_latency[-1]),
    xytext=(tokens[-1] - 50, cuda_latency[-1] * 3),
    arrowprops=dict(arrowstyle="->", color="black", lw=2),
    fontsize=12,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
)

plt.tight_layout()
plt.show()
