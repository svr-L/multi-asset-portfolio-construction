from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, figsize: tuple[int, int] = (10, 8)) -> None:
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_risk_return_scatter(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    annotate: bool = True,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c="white", edgecolors="red")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if annotate:
        for xi, yi, label in zip(x, y, x.index):
            plt.text(xi, yi, label, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_weights(weights: pd.Series, title: str) -> None:
    plt.figure(figsize=(10, 7))
    plt.pie(weights, labels=weights.index, autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.show()
