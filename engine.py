import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
from PIL import Image

from pydantic import BaseModel
from typing import List
import json

import streamlit as st
import datetime
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

import torch

from tqdm import tqdm
from glob import glob
from queue import Queue
import random

from clip_wrapper import MODEL_DIM, IMAGENET_CLASSES, IMAGENET_TEMPLATES
from clip_wrapper import ClipWrapper
from face_wrapper import FaceProcessor

DATASET_PATH = "dataset.parquet"
KM_THRESH = 50
FACES_THRESHOLD = 0.70


class SearchResult(BaseModel):
    id: str
    format: str
    image_path: str
    timestamp: float
    score: float
    date: datetime.date
    location: str

class SemanticSearcher(object):
    def __init__(self, dataset):
        
        self.dim_columns = [f"dim_{idx}" for idx in range(MODEL_DIM)]
        self.embedder = ClipWrapper().texts2vec
        self.imposter = ClipWrapper().images2vec
        
        self.dataset = dataset
    
    def search(self, query: str, indices: List = None, num_frames: int = 16):

        if not isinstance(indices, list):
            indices = list(range(len(self.dataset)))

        vector = self.embedder([query]).detach().numpy()
        dataset = self.dataset.iloc[indices].reset_index(drop=True)

        image_matrix = dataset[self.dim_columns].values

        logits = 100. * torch.tensor(image_matrix) @ torch.tensor(vector).T
        logits = logits.squeeze()

        similarity = logits.topk(min(len(indices), num_frames))

        return [
            SearchResult(
                id=row["id"],
                format=row['format'],
                image_path=row["image_path"],
                timestamp=row["timestamp"],
                score=score,
                date=datetime.date.today(),
                location="0., 0."
            )
            for score, (_, row) in zip(similarity[0].numpy(), dataset.iloc[similarity[1].numpy()].iterrows())
        ]
    
    def similar(self, image: Image, indices: List = None, num_frames: int = 16):
        
        if not isinstance(indices, list):
            indices = list(range(len(self.dataset)))
        
        vector = self.imposter([image]).detach().numpy()
        dataset = self.dataset.iloc[indices].reset_index(drop=True)

        image_matrix = dataset[self.dim_columns].values

        logits = 100. * torch.tensor(image_matrix) @ torch.tensor(vector).T
        logits = logits.squeeze()

        similarity = logits.topk(min(len(indices), num_frames))
        
        return [
            SearchResult(
                id=row["id"],
                format=row['format'],
                image_path=row["image_path"],
                timestamp=row["timestamp"],
                score=score,
                date=datetime.date.today(),
                location="0., 0."
            )
            for score, (_, row) in zip(similarity[0].numpy(), dataset.iloc[similarity[1].numpy()].iterrows())
        ]


class ImageClassifier(object):
    def __init__(self, dataset):

        dim_columns = [f"dim_{idx}" for idx in range(MODEL_DIM)]

        zeroshot_weights = ClipWrapper().generate_zero_shot_weights()
        image_matrix = torch.tensor(dataset[dim_columns].values)
        
        logits = 100. * image_matrix @ zeroshot_weights
        preds = logits.topk(1, 1, True, True) # Get the first prediction only

        dataset['category'] = preds[1].squeeze().numpy()
        dataset['confidence'] = preds[0].squeeze().numpy()

        dataset['category'] = dataset['category'].apply(lambda x: IMAGENET_CLASSES[x])

        self.dataset = dataset.drop(columns=dim_columns)
    
    def possible_classes(self, confidence: float):

        dataset = self.dataset[self.dataset['confidence'] >= confidence]
        mapping = dataset['category'].value_counts()
        
        classes = mapping.index.values.tolist()
        nums = mapping.values.tolist()

        return dict(zip(classes, nums))
    
    def get_class_indices(self, classes: List, confidence: float):
        dataset = self.dataset[self.dataset['confidence'] >= confidence]
        return dataset[dataset['category'].isin(classes)].index.values.tolist()
    
    def fetch(self, indices: List, num_frames: int = 16):
        dataset = self.dataset.iloc[indices]
        return [
            SearchResult(
                id=row["id"],
                format=row['format'],
                image_path=row["image_path"],
                timestamp=row["timestamp"],
                score=-1.,
                date=datetime.date.today(),
                location="0., 0."
            )
            for _, row in dataset.iterrows()
        ][:num_frames]

    def __len__(self):
        return len(self.dataset)


class DateLocationFilterer(object):
    def __init__(self, dataset):
        self.dataset = dataset

        self.geolocator = Nominatim(user_agent="http")
    
    def latlong_address(self, latlong: str):
        return self.geolocator.reverse(latlong).address

    def address_latlong(self, address: str):
        coded = self.geolocator.geocode(address)
        return f"{coded.latitude}, {coded.longitude}"
    
    def check_minimum_date(self, indices: List = None):
        if not isinstance(indices, list):
            indices = list(range(len(self.dataset)))

        dataset = self.dataset.iloc[indices].reset_index(drop=True)
        metadata = [json.loads(element) for element in dataset['metadata'].values]

        values = []
        for idx, meta in enumerate(metadata):
            if meta['datetime']:
                value = datetime.datetime.strptime(meta['datetime'], "%Y:%m:%d %H:%M:%S").date()
                values.append(value)

        return min(values)
    
    def elements_with_dates(self, indices: List = None,
                            start_date: datetime.datetime = None, end_date: datetime.datetime = None,
                            num_frames: int = 16):

        if not isinstance(indices, list):
            indices = list(range(len(self.dataset)))

        dataset = self.dataset.iloc[indices].reset_index(drop=True)
        metadata = [json.loads(element) for element in dataset['metadata'].values]

        indices = []
        values = []
        for idx, meta in enumerate(metadata):
            if meta['datetime']:
                value = datetime.datetime.strptime(meta['datetime'], "%Y:%m:%d %H:%M:%S").date()
                if value >= start_date and value <= end_date:
                    indices.append(idx)
                    values.append(value)

        return [
            SearchResult(
                id=row["id"],
                format=row['format'],
                image_path=row["image_path"],
                timestamp=row["timestamp"],
                score=-1,
                date=value,
                location="0., 0."
            )
            for value, (_, row) in zip(values, dataset.iloc[indices].iterrows())
        ][:num_frames]
    
    def elements_with_locations(self, indices: List = None, source: str = None, num_frames: int = 16):
        
        if not isinstance(indices, list):
            indices = list(range(len(self.dataset)))

        dataset = self.dataset.iloc[indices].reset_index(drop=True)
        metadata = [json.loads(element) for element in dataset['metadata'].values]

        indices = []
        values = []
        for idx, meta in enumerate(metadata):
            if meta['latitude'] and meta['longitude']:
                
                value = f"{meta['latitude']}, {meta['longitude']}"

                if source:
                    if geodesic(eval(source), eval(value)).kilometers <= KM_THRESH:
                        indices.append(idx)
                        values.append(value)
                    
                else:
                    indices.append(idx)
                    values.append(value)
        
        return [
            SearchResult(
                id=row["id"],
                format=row['format'],
                image_path=row["image_path"],
                timestamp=row["timestamp"],
                score=-1,
                date=datetime.date.today(),
                location=value
            )
            for value, (_, row) in zip(values, dataset.iloc[indices].iterrows())
        ][:num_frames]

class FaceClassifier(object):
    def __init__(self, images, filter_height: int = 60, filter_width: int = 60):
        self.face_processor = FaceProcessor(images)
        self.face_processor.filter(filter_height, filter_width)
        self.face_processor.calculate_embeddings()

        self.embeddings = self.face_processor.embeddings
        self.clusters = self.cluster_faces()
        self.clusters = sorted(self.clusters, key=lambda l: (len(l), l), reverse=True)
    
    def cosine(self, embedding1: np.ndarray, embedding2: np.ndarray):
        return torch.cosine_similarity(
            torch.tensor(embedding1).unsqueeze(0),
            torch.tensor(embedding2).unsqueeze(0)
        ).item()

    def cluster_faces(self):
        """
        THis function is written by OJU
        """
        list_of_clusters=[]
        dic_of_elements_covered={}
        graph={}

        for emb in tqdm(range(len(self.embeddings)), desc='generating face graph'):
            graph[emb] = []

            for y in range(emb+1, len(self.embeddings)):
                if self.cosine(self.embeddings[emb], self.embeddings[y]) >= FACES_THRESHOLD:
                    graph[emb].append(y)

        for emb in tqdm(range(len(self.embeddings)), desc='creating clusters'):
            lst=[]
            if emb in dic_of_elements_covered: continue
            
            q=Queue()
            q.put(emb)

            while (not q.empty()):
                curr=q.get()
                lst.append(curr)
                dic_of_elements_covered[curr]=None
                for x in graph[curr]:
                    if x in dic_of_elements_covered: continue
                    q.put(x)

            list_of_clusters.append(lst)
        
        return list_of_clusters

    def fetch_top_faces(self, num_faces: int = 10):
        clusters = self.clusters[:num_faces]

        for cluster in clusters:
            index = random.choice(cluster)
            path = self.face_processor.filepaths[index]

            yield path, len(cluster)

@st.cache_resource
def get_semantic_searcher():
    return SemanticSearcher(pd.read_parquet(DATASET_PATH))

@st.cache_resource
def get_image_classifier():
    return ImageClassifier(pd.read_parquet(DATASET_PATH))

@st.cache_resource
def get_datelocation_filterer():
    return DateLocationFilterer(pd.read_parquet(DATASET_PATH))

@st.cache_resource
def get_face_classifier():
    return FaceClassifier(glob("images/FACE*.jpeg"))

def execute():
    st.title("AI Demo")
    st.text("Setting Up Dependencies... Running Script for the first time might take a minute")

    searcher = get_semantic_searcher()
    classifier = get_image_classifier()
    filterer = get_datelocation_filterer()
    facer = get_face_classifier()

    confidence = st.slider("Choose a confidence value, (Default: 30)", 0, 100, 30)
    possible_classes = classifier.possible_classes(confidence=confidence)
    classes = st.multiselect(
        label="Choose Categories", 
        options= [f'ALL: {len(classifier)}'] + [f"{key}: {value}" for key, value in possible_classes.items()], 
        default=f'ALL: {len(classifier)}')
    
    if len(classes) == 0:
        st.error('Please choose atleast 1 class', icon="🚨")
    
    classes = list(map(lambda x: x.split(":")[0], classes))
    if "ALL" in classes:
        choose_indices = list(range(len(classifier)))
    else:
        choose_indices = classifier.get_class_indices(classes, confidence)
    
    for result in classifier.fetch(choose_indices):
        image = Image.open(os.path.join("images/", result.image_path))
        image.thumbnail((256, 256))
        st.image(image)

    query = st.text_input("Type in anything you want to search within Images/Videos")
    if query:
        for result in searcher.search(query, choose_indices):
            if result.score >= 25.:
                image = Image.open(os.path.join("images", result.image_path))
                image.thumbnail((256, 256))
                st.image(image)
                st.markdown(f"Format={result.format} Timestamp={result.timestamp} Confidence={result.score}")
    
    
    uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
    if uploadFile:
        image = Image.open(uploadFile)
        for result in searcher.similar(image, choose_indices):
            if result.score >= 75.:
                image = Image.open(os.path.join("images", result.image_path))
                image.thumbnail((256, 256))
                st.image(image)
                st.markdown(f"Format={result.format} Timestamp={result.timestamp} Confidence={result.score}")
    
    start_date = st.date_input("Start Date", min_value=filterer.check_minimum_date(), 
                               value=pd.to_datetime("2008-01-31", format="%Y-%m-%d"))
    end_date = st.date_input("End Date", min_value=filterer.check_minimum_date(),
                              value=pd.to_datetime("today", format="%Y-%m-%d"))

    for result in filterer.elements_with_dates(choose_indices, start_date=start_date, end_date=end_date):
        image = Image.open(os.path.join("images", result.image_path))
        image.thumbnail((256, 256))
        st.image(image)
        st.markdown(f"{result.date}")

    location = st.text_input("Type in a place you remember taking a photo/video")
    if location:
        latlong = filterer.address_latlong(location)
    else:
        latlong = None

    for result in filterer.elements_with_locations(indices=choose_indices, source=latlong):
        image = Image.open(os.path.join("images", result.image_path))
        image.thumbnail((256, 256))
        st.image(image)
        st.markdown(f"{filterer.latlong_address(result.location)}")
    
    st.text("Some unique people we found througout")
    for face_path, occurance in facer.fetch_top_faces():
        image = Image.open(face_path)
        image.thumbnail((60, 60))
        st.image(image)
        st.markdown(f"occuranced {occurance} times")



if __name__ == "__main__":
    execute()