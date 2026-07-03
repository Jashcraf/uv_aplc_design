import numpy as np
import matplotlib.pyplot as plt
import hwostyle
hwostyle.use("light")

GPUs = ["Ryzen 9 7900X",
        "M1",
        "RTX 4090",
        "V100",
        "A100",
        "GH-200"]
mean_speed_s = [2.97, 6.15, 38.7, 37.83, 49.13, 0]

# Approximate theoretical peak FP64 (double-precision) throughput, in TFLOPS
fp64_tflops = [0.9, 0.16, 1.29, 7.8, 9.7, 34.0]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
bar_width = 0.6
ax.bar(GPUs, mean_speed_s, width=bar_width)
ax.set_xticklabels(GPUs, rotation=45, ha="right")
ax.set_ylabel("L-BFGS-B Iteration / s")
ax.set_xlabel("Processor Architecture")

# Dotted box around the first three bars
box_x0 = -0.5
box_x1 = 2.5
box_y1 = max(mean_speed_s[:3]) * 1.15
box = plt.Rectangle((box_x0, 0), box_x1 - box_x0, box_y1,
                    fill=False, linestyle=":", edgecolor="black", linewidth=1.5)
ax.add_patch(box)
ax.text((box_x0 + box_x1) / 2, box_y1, "Consumer hardware",
        ha="center", va="bottom")

# Second subplot: theoretical peak FP64 performance
ax2.bar(GPUs, fp64_tflops, width=bar_width)
ax2.set_xticklabels(GPUs, rotation=45, ha="right")
ax2.set_ylabel("Theoretical Peak FP64 Performance / TFLOPS")
ax2.set_xlabel("Processor Architecture")

fig.tight_layout()
plt.show()
