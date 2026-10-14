"""
Generate Figure 1: Application Context — Coffee Shop Demand Characteristics
Multi-panel figure for ESWA submission (300 DPI, Times New Roman, serif)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/Users/bytedance/Documents/门店预测kaggle数据/draft_e7b169ef_folder/figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Load computed statistics
data = np.load(str(OUT / "data_context.npz"))
zero_rates = data["zero_rates"]
adi_finite = data["adi_vals"][np.isfinite(data["adi_vals"])]
mean_demand = data["mean_demand"]

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "light": "#D6EAF8",
    "dark": "#1a5276",
}

# ============================================================
# Figure 1: 2x2 Multi-panel Application Context
# ============================================================
fig = plt.figure(figsize=(10, 7.5))

# ---- (a) Sample SKU time series ----
ax1 = fig.add_subplot(2, 2, 1)
import pandas as pd
df = pd.read_excel("/Users/bytedance/Documents/门店预测kaggle数据/Coffee Shop Sales.xlsx")
df["date"] = pd.to_datetime(df["transaction_date"])
daily = df.groupby(["date", "store_location", "product_detail"])["transaction_qty"].sum().reset_index()
pivot = daily.pivot_table(index="date", columns=["store_location", "product_detail"],
                           values="transaction_qty", fill_value=0)

# Select representative SKUs: low, medium, high demand
col_means = pivot.mean()
low_idx = col_means.nsmallest(3).index[-1]
mid_idx = col_means.nlargest(int(len(col_means) * 0.5)).nsmallest(1).index[0]
high_idx = col_means.nlargest(2).index[-1]

for idx, label, color, alpha_val in [
    (high_idx, "High-demand (Drip Coffee)", COLORS["primary"], 1.0),
    (mid_idx, "Medium-demand (Latte)", COLORS["accent"], 1.0),
    (low_idx, "Low-demand (Specialty Tea)", COLORS["secondary"], 1.0),
]:
    ts = pivot[idx].values
    # Smooth with 7-day rolling for trend
    ax1.plot(pivot.index, ts, alpha=0.3, color=color, linewidth=0.8)
    smoothed = pd.Series(ts).rolling(7, center=True).mean()
    ax1.plot(pivot.index, smoothed, color=color, linewidth=1.5, label=label)

ax1.set_ylabel("Daily Sales Quantity")
ax1.set_title("(a) Representative SKU Time Series (Jan–Jun 2023)", fontweight="bold", loc="left")
ax1.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=7)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(axis="x", rotation=30)

# ---- (b) Zero-rate histogram ----
ax2 = fig.add_subplot(2, 2, 2)
bins = np.linspace(0, 1, 31)
ax2.hist(zero_rates, bins=bins, color=COLORS["primary"], edgecolor="white", alpha=0.85, linewidth=0.5)
ax2.axvline(np.mean(zero_rates), color=COLORS["secondary"], linestyle="--", linewidth=1.5,
            label=f"Mean = {np.mean(zero_rates):.1%}")
ax2.axvline(np.median(zero_rates), color=COLORS["accent"], linestyle=":", linewidth=1.5,
            label=f"Median = {np.median(zero_rates):.1%}")
ax2.set_xlabel("Zero-Sales Proportion")
ax2.set_ylabel("Number of SKU-Store Pairs")
ax2.set_title("(b) Zero-Sales Rate Distribution", fontweight="bold", loc="left")
ax2.legend(fontsize=7)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Add annotation
ax2.annotate(f"{(zero_rates > 0.5).sum()} SKUs\nhave >50% zero days",
             xy=(0.75, 0.8), xycoords="axes fraction",
             fontsize=8, ha="center", color=COLORS["dark"],
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"], alpha=0.8))

# ---- (c) ADI Distribution ----
ax3 = fig.add_subplot(2, 2, 3)
adi_clipped = np.clip(adi_finite, 1, 30)
bins = np.arange(1, 32, 1)
ax3.hist(adi_clipped, bins=bins, color=COLORS["secondary"], edgecolor="white", alpha=0.85, linewidth=0.5)
ax3.axvline(np.mean(adi_clipped), color=COLORS["primary"], linestyle="--", linewidth=1.5,
            label=f"Mean ADI = {np.mean(adi_clipped):.1f}")
ax3.axvline(np.median(adi_clipped), color=COLORS["accent"], linestyle=":", linewidth=1.5,
            label=f"Median ADI = {np.median(adi_clipped):.1f}")
ax3.set_xlabel("Average Demand Interval (days)")
ax3.set_ylabel("Number of SKU-Store Pairs")
ax3.set_title("(c) Average Demand Interval Distribution", fontweight="bold", loc="left")
ax3.legend(fontsize=7)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ---- (d) Store Location Comparison ----
ax4 = fig.add_subplot(2, 2, 4)
store_daily = df.groupby(["date", "store_location"])["transaction_qty"].sum().reset_index()
store_pivot = store_daily.pivot_table(index="date", columns="store_location",
                                       values="transaction_qty", fill_value=0)
store_stats = []
for loc in store_pivot.columns:
    vals = store_pivot[loc].values
    store_stats.append({
        "location": loc,
        "avg_daily": vals.mean(),
        "total": vals.sum(),
        "zero_days": (vals == 0).mean(),
        "cv": vals.std() / vals.mean() if vals.mean() > 0 else 0,
    })
store_stats.sort(key=lambda x: x["avg_daily"], reverse=True)

locations = [s["location"] for s in store_stats]
avg_daily = [s["avg_daily"] for s in store_stats]
zeros = [s["zero_days"] * 100 for s in store_stats]

bars = ax4.bar(range(len(locations)), avg_daily, color=COLORS["primary"], edgecolor="white",
               width=0.55, label="Avg Daily Sales")
ax4_twin = ax4.twinx()
ax4_twin.plot(range(len(locations)), zeros, "D-", color=COLORS["secondary"], linewidth=2,
              markersize=8, markerfacecolor="white", markeredgewidth=1.5, label="Zero-Day %")

ax4.set_xticks(range(len(locations)))
ax4.set_xticklabels(locations, rotation=15, ha="right", fontsize=7)
ax4.set_ylabel("Average Daily Sales", color=COLORS["primary"])
ax4_twin.set_ylabel("Zero-Day Rate (%)", color=COLORS["secondary"])
ax4.set_title("(d) Store-Level Comparison", fontweight="bold", loc="left")
ax4.spines["top"].set_visible(False)

# Create combined legend
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7,
           frameon=True, fancybox=True, framealpha=0.9)

fig.suptitle("Figure 1: Coffee Shop Demand Characteristics — Application Context",
             fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig1_application_context.png", bbox_inches="tight")
plt.close(fig)
print("Figure 1 saved: fig1_application_context.png")

# Print key stats for the paper
print(f"\nKey statistics for the paper:")
print(f"  Total SKU-store pairs: {len(zero_rates)}")
print(f"  Date range: {pivot.index.min().strftime('%Y-%m-%d')} to {pivot.index.max().strftime('%Y-%m-%d')} ({len(pivot)} days)")
print(f"  Stores: {len(stores) if 'stores' in dir() else 'N/A'}")
print(f"  Zero-rate: mean={np.mean(zero_rates):.1%}, median={np.median(zero_rates):.1%}")
print(f"  ADI: mean={np.mean(adi_finite):.1f}, median={np.median(adi_finite):.1f}")
print(f"  SKUs with >50% zero days: {(zero_rates > 0.5).sum()} / {len(zero_rates)} ({(zero_rates>0.5).mean()*100:.0f}%)")
print(f"  Mean daily demand: {np.mean(mean_demand):.3f} (median={np.median(mean_demand):.3f})")