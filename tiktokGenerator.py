import json
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import os
import tarfile
import datetime
import pafy
from torch.nn import *
from torch.optim import *

class VideoGeneratorModel(nn.Module):
    def __init__(self):
        super(VideoGeneratorModel, self).__init__()
        # Define layers of the generator here
        # ...

    def forward(self, x):
        # Perform forward pass through the model
        # ...
        
date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

def list_videos_in_directory(url):
    with tarfile.open(url, 'r:gz') as tar:
        tar.extractall()
        
    video_list = []
    for member in tar.getmembers():
        if member.name.endswith('.json'):
            with tar.extractfile(member) as file:
                data = json.load(file)
                for video in data.values():
                    video_list.append(video["url"])
    return video_list


def train_video_generator(model, dataset, epochs, learning_rate):
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(dataset):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = frame.transpose(2, 0, 1)
        frame = torch.from_numpy(frame)
        frame = frame.float()
        frames.append(frame)
    video.release()
    return frames

class KineticsDataset(Dataset):
    def __init__(self, url):
        self.url = url
        self.video_list = list_videos_in_directory(url)
        self.metadata = [{'video_url': video} for video in self.video_list]
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        video_path = self.metadata[idx]['video_path']
        video = load_video(video_path)
        return video, video[1:]


def generate_video(data_loader, video_path):
    fourcc = cv2.VideoWriter_fourcc("mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (224, 224), isColor=True)
    for _, (frames, _) in enumerate(data_loader):
        for frame in frames:
            frame = frame.numpy().transpose(1, 2, 0)
            frame = (frame*255).astype('uint8')
            video_writer.write(frame)
    video_writer.release()
    print(f"Video generated at {video_path}")


# Creazione del modello di generazione video
video_generator_model = VideoGeneratorModel()

url = "kinetics700_2020.tar.gz"
kinetics_dataset = KineticsDataset(url)
data_loader = DataLoader(kinetics_dataset, batch_size=32, shuffle=True)

# Addestramento del modello utilizzando la funzione train_video_generator
train_video_generator(video_generator_model, kinetics_dataset, epochs=10, learning_rate=0.001)

# Generazione del video utilizzando il modello addestrato
video_path = f"generated_video_{date}.mp4"
generate_video(data_loader, video_path)
