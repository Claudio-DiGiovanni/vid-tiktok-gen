import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import os
import tarfile

def list_videos_in_directory(data_dir):
    videos = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            with open(os.path.join(data_dir, file)) as f:
                for line in f:
                    videos.append(line.strip())
    return videos


def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (224, 224)))
    video.release()
    return frames

class KineticsDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.metadata = pd.read_json(os.path.join(data_dir, "train.json"))
        self.videos = self.metadata


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.videos.iloc[idx]["filename"])
        video = load_video(video_path)
        return video


def generate_video(data_loader, video_path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (224, 224), isColor=True)
    for batch in data_loader:
        for frame in batch:
            video_writer.write(frame)

    video_writer.release()
    print("Video generated at {}".format(video_path))

# decomprimere il file .tar.gz
with tarfile.open("kinetics700_2020.tar.gz", "r:gz") as tar:
    tar.extractall()

# utilizzare la cartella decompressa come percorso per data_dir
data_dir = "kinetics700_2020"
kinetics_dataset = KineticsDataset(data_dir)
data_loader = DataLoader(kinetics_dataset, batch_size=32, shuffle=True)
video_path = "generated_video.mp4"
generate_video(data_loader, video_path)