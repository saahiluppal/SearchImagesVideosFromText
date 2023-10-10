# SearchImagesVideosFromText
semantic search through videos and images through text

# How to Use
1. Install requirements using ```pip install -r requirements.txt```
2. Create a folder named ```raw_data``` and populate the folder with the images and videos you want to search within.
3. execute python file ```process_raw_data.py``` to process to save the vectors
4. execute python file ```process_parquets.py``` to combine the saved vectors
5. run the demo through ```streamlit run engine.py``` and go to 127.0.0.1:8000 for the demo
