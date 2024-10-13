from scipy import signal
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as df

def preprocessing(signal_data):
    nperseg_ = 100
    if nperseg_ >= len(signal_data):
        raise ValueError(f"Array must contain at least {nperseg_} samples")
    filter_but = signal.butter(
        N=5, Wn=[1.0, 15], btype="bandpass", fs=100, output="sos"
    )
    filtered_signal = signal.sosfiltfilt(filter_but, signal_data)
    return filtered_signal


class CustomTrace2DDataset(Dataset):
    def __init__(self, data_info_pd, W=8192, n_classes=6, N=256):
        self.data_info = data_info_pd
        self.W = W
        self.n_classes = n_classes
        self.N = N

    def __len__(self):
        return len(self.data_info["true_label"])

    def __getitem__(self, idx):
        path = self.data_info.loc[idx]["event_path"]
        event_name = self.data_info.loc[idx]["event_name"]
        data_input = np.load(path)
        data_input = torch.tensor(data_input.copy())
        X = data_input[:8].float()
        y = data_input[8 : 9 + self.n_classes].float()
        patches = X.unfold(1, self.N, self.N)
        patches = patches.permute(1, 0, 2)
        X = patches.reshape(-1, self.N).unsqueeze(0)
        patches = y.repeat(8, 1, 1)
        patches = patches.permute(1, 0, 2)
        patches = patches.unfold(2, self.N, self.N)
        patches = patches.permute(0, 2, 1, 3)
        y = patches.reshape(self.n_classes + 1, -1, self.N)
        return (
            X,
            y,
            event_name,
        )


def print_trace(
    trace_data,
    n_stations=8,
    save_path=None,
    save=False,
    title="",
    num=0,
    dpi=100,
    figsize=(10, 6),
    font_size=10,
):
    colors = {
        "input": "#4C72B0",
        0: "#808080",
        1: "#df8d5e",
        2: "#2ca02c",
        3: "#d62728",
        4: "#9467bd",
        5: "#8c564b",
    }
    labels = {
        1: "station 1",
        2: "station 2",
        3: "station 3",
        4: "station 4",
        5: "station 5",
        6: "station 6",
        7: "station 7",
        8: "station 8",
        9:  "BG",
        10: "VT",
        11: "LP",
        12: "TR",
        13: "AV",
        14: "IC",
    }
    matplotlib.rcParams.update({"font.size": font_size})

    fig, axes = plt.subplots(
        n_stations + 6, 1, sharex=True, figsize=figsize, num=num, dpi=dpi
    )  
    for idx, wave in enumerate(trace_data):
        if idx < n_stations:
            sns.lineplot(
                x=np.arange(trace_data.shape[1]) / 100,
                y=wave,
                ax=axes[idx],
                color=colors["input"],
                lw=0.8,
            )
            axes[idx].set_ylim(-1.2, 1.2)  
        else:
            sns.lineplot(
                x=np.arange(trace_data.shape[1]) / 100,
                y=wave,
                ax=axes[idx],
                color=colors[idx - n_stations],
                lw=2,
            )
            axes[idx].set_ylim(-.1, 1.1) 
        axes[idx].set_ylabel(f"                {labels[idx + 1]}", rotation=0)
        axes[idx].yaxis.set_label_position("right")  # Place y-labels on the right
    axes[-1].set_xlabel("time [s]")  
    if save:
        plt.savefig(save_path)
        plt.close("all")
    else:
        plt.suptitle(title)


def fold_X(X, N=256):
    patches = X.unfold(1, N, N)
    patches = patches.permute(1, 0, 2)
    X = patches.reshape(-1, N).unsqueeze(0)
    return X.float()


def fold_y(y, N=256, n_classes=6, n_stations=8):
    patches = y.repeat(n_stations, 1, 1)
    patches = patches.permute(1, 0, 2)
    patches = patches.unfold(2, N, N)
    patches = patches.permute(0, 2, 1, 3)
    y = patches.reshape(n_classes, -1, N)
    return y.float()


def unfold_y(img, W=8192, N=256, n_classes=6):
    output = torch.zeros([len(img), n_classes, W])
    for idx, patches in enumerate(img):
        patches = patches.unfold(1, 8, 8)
        patches = patches.permute(0, 3, 1, 2).reshape(n_classes, 8, N * N // 8)
        patches_y = patches.sum(axis=1)
        patches_y = patches_y / patches_y.max()
        output[idx] = patches_y
    del patches_y, img
    return output.float()


def unfold_X(img, W=8192):
    output = torch.zeros([len(img), 8, W])
    for idx, patches in enumerate(img):
        patches = patches.squeeze(0)
        patches = patches.unfold(0, 8, 8)
        patches = torch.cat(patches.unbind(dim=0))
        patches = patches.permute(1, 0)
        output[idx] = patches
    del patches, img
    return output.float()

def detected_events(BG_diff):
    start_indices = np.where(BG_diff == -1)[0]
    if len(start_indices) == 0:
        start_indices = np.array([0])
    end_indices = np.where(BG_diff == 1)[0]
    if len(end_indices) == 0:
        end_indices = np.array([-1])
    events = []
    last_end_idx = -1
    for start in start_indices:
        valid_ends = end_indices[end_indices > start]
        if valid_ends.size > 0:
            end = valid_ends[0]
            events.append((start, end, end - start))
            last_end_idx = end
        else:
            events.append([start, len(BG_diff) - 1, len(BG_diff) - 1 - start])
    if last_end_idx != -1:
        invalid_ends = end_indices[end_indices < start_indices[0]]
        for invalid_end in invalid_ends:
            events.insert(0, (0, invalid_end, invalid_end))
    events_df = df.DataFrame(events, columns=["start", "end", "length"])
    return events_df


def post_processing(
    output,
    clases_OVDAS={
        0.0: "BG",
        1.0: "VT",
        2.0: "LP",
        3.0: "TR",
        4.0: "AV",
        5.0: "IC",
    },
):
    max_indices = np.argmax(output, axis=0)
    processed_out = np.eye(len(output))[max_indices].T
    BG_diff = np.diff(processed_out[0])
    if np.abs(BG_diff).sum() != 0:
        events_df = detected_events(BG_diff)
        events_df["class_n"] = None
        events_df["class_label"] = None
        for index, row in events_df.iterrows():
            start_ = row["start"]
            end_ = row["end"]
            length = row["length"]
            predicted_class = processed_out[1:, start_:end_].sum(axis=1).argmax() + 1
            pred_label = clases_OVDAS[predicted_class]
            events_df.at[index, "class_n"] = predicted_class
            events_df.at[index, "class_label"] = pred_label
    else:
        events_df = df.DataFrame(
            [[0, 8191, 8192, None, None]],
            columns=["start", "end", "length", "class_n", "class_label"],
        )
        predicted_class = processed_out.sum(axis=1).argmax()
        pred_label = clases_OVDAS[predicted_class]
        events_df.at[0, "class_n"] = predicted_class
        events_df.at[0, "class_label"] = pred_label
    return events_df



def plot_segmentation(
    input_traces,
    detected_events_df,
    save_path=None,
    save=False,
    title="",
    num=0,
    dpi=100,
    figsize=(10, 6),
):
    colors = {
        "input": "#4C72B0",
        0: "#808080",
        1: "#df8d5e",
        2: "#2ca02c",
        3: "#d62728",
        4: "#9467bd",
        5: "#8c564b",
    }
    labels = {
        1: "FRE",
        2: "SHG",
        3: "NBL",
        4: "CHA",
        5: "FU2",
        6: "CHS",
        7: "LBN",
        8: "PLA",
        9: "BG",
        10: "VT",
        11: "LP",
        12: "TR",
        13: "AV",
        14: "IC",
    }

    fig, axes = plt.subplots(
        8, 1, sharex=True, figsize=figsize, num=num, dpi=dpi
    )  # Increased height to 12 for better spacing
    for idx, wave in enumerate(input_traces):
        if idx <= 7:
            sns.lineplot(
                x=np.arange(input_traces.shape[1]) / 100,
                y=wave,
                ax=axes[idx],
                color=colors["input"],
                lw=1,
            )
        # Set custom y-label
        axes[idx].set_ylabel(f"           {labels[idx + 1]}", rotation=0)
        # axes[idx].set_yticks([])  # Remove y-axis ticks

        axes[-1].set_xlabel("time [s]")  # Shared x-axis label
        axes[idx].yaxis.set_label_position("right")  # Place y-labels on the left
        axes[idx].set_ylim(-1.2, 1.2)  # Set limits for y-axis
    #detected_events_df = detected_events_df.query("length>250")
    for index, row in detected_events_df.iterrows():
        for idx, wave in enumerate(input_traces):
            axes[idx].axvspan(
                row["start"] / 100,
                row["end"] / 100,
                color=colors[row["class_n"]],
                alpha=0.3,
            )
        midpoint = (row["start"] + row["end"]) / 200  # Midpoint of the event
        plt.text(
            midpoint,
            1.05,
            row["class_label"],
            ha="center",
            va="bottom",
            transform=axes[0].get_xaxis_transform(),
            fontsize=12,
        )

    if save:
        plt.savefig(save_path)
        plt.close("all")
    else:
        plt.suptitle(title)
        # plt.show()