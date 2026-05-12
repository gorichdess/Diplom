import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def parse_log(filepath):
    pattern = (r"UPD\s+(\d+)\s*\|\s*EP\s+(\d+)\s*\|\s*Avg100\s+([\d\-\.]+)\s*\|\s*AvgSteps\s+([\d\-\.]+)\s*\|\s*Succ\s+([\d\.]+)%\s*\|\s*Loss\s+([\d\-\.]+)\s*\|\s*Diff\s+([\d\.]+)\s*\|\s*Ent\s+([\d\-\.]+)\s*\|\s*LR\s+([\d\-\.eE]+)\s*\|\s*Time\s+([\d\.]+)")
    data = []
    with open(filepath, "r") as f:
        for line in f:
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
    df = pd.DataFrame(data, columns=["upd", "ep", "avg100", "avg_steps", "succ", "loss", "diff", "ent", "lr", "time"])
    return df

def plot_metrics(df_dict, save_prefix="training_plot"):
    """df_dict: словник {label: DataFrame} для кількох запусків"""
    metrics = [
        ("avg100", "Середня винагорода (Avg100)", "Винагорода"),
        ("succ", "Успішність, %", "Успішність, %"),
        ("diff", "Складність (Difficulty)", "Складність"),
        ("loss", "Функція втрат", "Loss"),
        ("ent", "Ентропія", "Entropy"),
        ("lr", "Швидкість навчання", "Learning rate"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (col, title, ylabel) in enumerate(metrics):
        ax = axes[i]
        for label, df in df_dict.items():
            ax.plot(df["upd"], df[col], linewidth=0.8, label=label)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Номер оновлення PPO")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    # Прибираємо зайву пусту піддіаграму, якщо 2x3, а метрик 6 — все заповнено
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=200)
    plt.savefig(f"{save_prefix}.pdf")
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
        plot_metrics(df_dict, save_prefix=args.output)
    else:
        print("Немає даних для візуалізації.")