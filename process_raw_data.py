import cv2
from tqdm import tqdm
import uuid

from pathlib import Path
from PIL import Image
import pandas as pd

from clip_wrapper import MODEL_DIM
from clip_wrapper import ClipWrapper
clip_wrapper = ClipWrapper()

from face_wrapper import FaceWrapper
face_wrapper = FaceWrapper()

from image_wrapper import ImageWrapper
image_wrapper = ImageWrapper()


FRAME_EXTRACT_RATE_SECONDS = 1
IMAGE_MAX_DIM_SAVE = (512, 512)

DATAFRAME_PATH = Path("parquet-data/")
DATAFRAME_PATH.mkdir(exist_ok=True)

IMAGES_PATH = Path('images/')
IMAGES_PATH.mkdir(exist_ok=True)

videos = Path("raw_data").glob("*.mp4")
images = Path("raw_data").glob("*.jpg")


def get_clip_vectors_video(video_path, clip_wrapper):
    
    cap = cv2.VideoCapture(str(video_path))
    num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    extract_every_n_frames = int(FRAME_EXTRACT_RATE_SECONDS * fps)
    
    for frame_idx in tqdm(range(num_video_frames), desc=f"Running CLIP on video {video_path.stem}"):
        ret, frame = cap.read()
        if frame_idx % extract_every_n_frames != 0: continue
        
        faces = list(face_wrapper.get_mtcnn_faces(frame[..., ::-1]))
        image = Image.fromarray(frame[..., ::-1])
        clip_vector = clip_wrapper.images2vec([image]).squeeze().numpy()
        timestamp_secs = frame_idx / fps
        
        yield clip_vector, faces, image, timestamp_secs, frame_idx
    
    cap.release()

def get_clip_vectors_image(image_path, clip_wrapper):
    
    frame = cv2.imread(str(image_path))
    faces = list(face_wrapper.get_mtcnn_faces(frame[..., ::-1]))

    image = Image.fromarray(frame[..., ::-1])
    clip_vector = clip_wrapper.images2vec([image]).squeeze().numpy()
    
    yield clip_vector, faces, image



def process_videos():
    
    for idx, video_path in enumerate(videos):

        video_id = video_path.stem
        complete_file = DATAFRAME_PATH.joinpath(video_id + ".parquet")
        if complete_file.exists(): print(complete_file, 'already exists;'); continue
        
        results = []
        for clip_vector, faces, image, timestamp_secs, frame_idx in get_clip_vectors_video(video_path, clip_wrapper):

            face_entropy = uuid.uuid4()
            for idx, face in enumerate(faces):
                face_path = IMAGES_PATH.joinpath(f"FACE-{video_id}-{face_entropy}-{idx}.jpeg")
                cv2.imwrite(str(face_path), face[..., ::-1])
            
            image_path = IMAGES_PATH.joinpath(f"{video_id}-{uuid.uuid4()}.jpeg")
            image.thumbnail(IMAGE_MAX_DIM_SAVE)
            image.save(image_path)
            
            results.append(
                [
                    "video",
                    video_id,
                    image_path.name,
                    f"FACE-{video_id}-{face_entropy}",
                    len(faces),
                    frame_idx,
                    timestamp_secs,
                    *clip_vector,
                ]
            )
    
        df = pd.DataFrame(
            results,
            columns=["format", "id", "image_path", "face_path", "num_faces", "frame_idx", "timestamp"]
            + [f"dim_{i}" for i in range(MODEL_DIM)],
        )

        print(f"Saving data to {complete_file}")
        df.to_parquet(complete_file, index=False)


def process_images():
    
    for idx, image_path in enumerate(tqdm(images)):

        image_id = image_path.stem
        complete_file = DATAFRAME_PATH.joinpath(image_id + ".parquet")
        if complete_file.exists(): print(complete_file, 'already exists;'); continue
        
        results = []
        for clip_vector, faces, image in get_clip_vectors_image(image_path, clip_wrapper):

            face_entropy = uuid.uuid4()
            for idx, face in enumerate(faces):
                face_path = IMAGES_PATH.joinpath(f"FACE-{image_id}-{face_entropy}-{idx}.jpeg")
                cv2.imwrite(str(face_path), face[..., ::-1])
            
            image_path = IMAGES_PATH.joinpath(f"{image_id}-{uuid.uuid4()}.jpeg")
            image.thumbnail(IMAGE_MAX_DIM_SAVE)
            image.save(image_path)
            
            results.append(
                [
                    "image",
                    image_id,
                    image_path.name,
                    f"FACE-{image_id}-{face_entropy}",
                    len(faces),
                    0,
                    0,
                    *clip_vector,
                ]
            )
    
        df = pd.DataFrame(
            results,
            columns=["format", "id", "image_path", "face_path", "num_faces", "frame_idx", "timestamp"]
            + [f"dim_{i}" for i in range(MODEL_DIM)],
        )

        # print(f"Saving data to {complete_file}")
        df.to_parquet(complete_file, index=False)

if __name__ == "__main__":
    process_videos()
    process_images()