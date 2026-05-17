# Script for parsing PPO training logs and visualizing learning dynamics.
# Used for experimental analysis in reinforcement learning evaluation.
import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

# Parse training log file generated during PPO training.
# Extracts metrics such as reward, success rate, loss, entropy, and curriculum difficulty.
def parse_log(filepath):
    # Regular expression for extracting structured PPO training metrics
    # from formatted log lines.
    pattern = (r"UPD\s+(\d+)\s*\|\s*EP\s+(\d+)\s*\|\s*Avg100\s+([\d\-\.]+)\s*\|\s*AvgSteps\s+([\d\-\.]+)\s*\|\s*Succ\s+([\d\.]+)%\s*\|\s*Loss\s+([\d\-\.]+)\s*\|\s*Diff\s+([\d\.]+)\s*\|\s*Ent\s+([\d\-\.]+)\s*\|\s*LR\s+([\d\-\.eE]+)\s*\|\s*Time\s+([\d\.]+)")
    
    # Container for parsed training samples (one per PPO update).
    data = []

    # Iterate through log file line-by-line and extract metrics.
    with open(filepath, "r") as f:
        for line in f:
            # Match structured log entry against expected PPO log format.
            match = re.search(pattern, line)
            if match:
                upd = int(match.group(1))
                ep = int(match.group(2))
                avg100 = float(match.group(3))
                avg_steps = float(match.group(4))
                succ = float(match.group(5))
                loss = float(match.group(6))
                diff = float(match.group(7))
                ent = float(match.group(8))
                lr = float(match.group(9))
                time = float(match.group(10))
                data.append([upd, ep, avg100, avg_steps, succ, loss, diff, ent, lr, time])

    # Convert parsed log data into structured tabular format
    # for statistical analysis and visualization.
    df = pd.DataFrame(data, columns=["upd", "ep", "avg100", "avg_steps", "succ", "loss", "diff", "ent", "lr", "time"])
    return df

# Visualize PPO training dynamics across multiple metrics.
# Used to evaluate learning stability and curriculum effect.
def plot_metrics(df_dict, save_prefix="training_plot"):
    # Key performance indicators of reinforcement learning training:
    # reward, success rate, difficulty, loss, entropy, learning rate.
    metrics = [
        ("avg100", "Середня винагорода (Avg100)", "Винагорода"),
        ("succ", "Успішність, %", "Успішність, %"),
        ("diff", "Складність (Difficulty)", "Складність"),
        ("loss", "Функція втрат", "Loss"),
        ("ent", "Ентропія", "Entropy"),
        ("lr", "Швидкість навчання", "Learning rate"),
    ]

    # Create multi-panel visualization of training metrics.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Plot metric evolution over PPO updates for each experiment.
    for i, (col, title, ylabel) in enumerate(metrics):
        ax = axes[i]
        for label, df in df_dict.items():
            # Visualize training curve for each experimental run.
            ax.plot(df["upd"], df[col], linewidth=0.8, label=label)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Номер оновлення PPO")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=200)
    plt.savefig(f"{save_prefix}.pdf")
    # Display plots interactively after generation.
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Побудова графіків з логів тренування")
    parser.add_argument("log_files", nargs="+", help="Шляхи до файлів логів")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Мітки для легенди (за замовчуванням імена файлів)")
    parser.add_argument("--output", default="training_curves", help="Базове ім'я для збереження графіків")
    args = parser.parse_args()

    df_dict = {}
    labels = args.labels if args.labels else [os.path.splitext(os.path.basename(f))[0] for f in args.log_files]
    for fpath, label in zip(args.log_files, labels):
        df = parse_log(fpath)
        if not df.empty:
            df_dict[label] = df
            print(f"Файл {fpath}: {len(df)} записів")
        else:
            print(f"Увага: не знайдено записів у {fpath}")

    if df_dict:
        # These plots are used to analyze:
        # - convergence speed of PPO
        # - effect of curriculum learning
        # - stability of policy optimization
        # - exploration behavior (entropy)
        # - adaptation of environment difficulty
        plot_metrics(df_dict, save_prefix=args.output)
    else:
        print("Немає даних для візуалізації.")