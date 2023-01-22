import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import os
import tarfile
import datetime

date = {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}

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
    print(f"Video generated at {date}".format(video_path))

# decomprimere il file .tar.gz
if not os.path.isdir("kinetics700_2020"):
    with tarfile.open("kinetics700_2020.tar.gz", "r:gz") as tar:
        tar.extractall()
    print("File decompresso con successo.")
else:
    print("La cartella kinetics700_2020 esiste già, non viene decompresso nuovamente.")

# utilizzare la cartella decompressa come percorso per data_dir
data_dir = "kinetics700_2020"
kinetics_dataset = KineticsDataset(data_dir)
data_loader = DataLoader(kinetics_dataset, batch_size=32, shuffle=True)
video_path = f"generated_video_{date}.mp4"
generate_video(data_loader, video_path)

# sistema il codice tenendo conto che i video da analizzare vengono forniti dal file .json che si ottiene quando si decomprime il file .tar.gz
# https://github.com/Claudio-DiGiovanni/vid-tiktok-gen