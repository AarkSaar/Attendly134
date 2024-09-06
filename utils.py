import os
import gspread
from google.oauth2.service_account import Credentials
import asyncio
import websockets
import json
import base64
import io as BytesIO
import numpy as np
import cv2 as cv
from datetime import datetime, timedelta
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from string import ascii_uppercase



print('hey')
def base64_to_opencv_image(base64_str):
    try:
        # Decode the base64 image
        img_data = base64.b64decode(base64_str.split(',')[1])
        # Convert bytes to numpy array
        np_array = np.frombuffer(img_data, np.uint8)
        # Decode numpy array to image
        image = cv.imdecode(np_array, cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image.")
        return image
    except Exception as e:
        print(f"Error in base64_to_opencv_image: {e}")
        return None


def get_column_header(sheet, value_to_find):
    # Find the cell containing the value
    cell = sheet.find(value_to_find)
    
    if cell:
        # Get the column index (1-based)
        column_index = cell.col
        
        # Convert column index to letter
        column_letter = ''
        while column_index:
            column_index, remainder = divmod(column_index - 1, 26)
            column_letter = ascii_uppercase[remainder] + column_letter
        
        return column_letter
    else:
        raise ValueError(f"Value '{value_to_find}' not found in the sheet.")
    

def extract_date_time(iso_string):
    # Correct format to handle the fractional second and 'Z' timezone indicator
    date_time = datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%S.%fZ")
    
    # Format the date and time separately if needed
    date = date_time.strftime("%d/%m/%Y")
    time = date_time.strftime('%H:%M:%S')
    
    return date, time



def addName(newInfo, sheet):
    # Find the last row with data in the column (e.g., column A)
    column_values = sheet.col_values(1)  # Fetch all values in column A
    last_row = len(column_values) + 1  # Get the index of the next row

    # Data to be added
    new_data = [[newInfo]]

    # Update the cell in the next available row in column A
    cell_range = f'A{last_row}'  # Adjust the column letter if needed
    try:
        sheet.update(cell_range, new_data)
        print(f"Added data to row {last_row} in column A.")
    except Exception as e:
        print(f"Error updating Google Sheet: {e}")


class FACELOADING:
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size
        self.detector = MTCNN()
    
    def resize_image(self, img):
        return cv.resize(img, self.target_size)

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        # Resize the image before face detection
        img = self.resize_image(img)
        
        # Detect faces
        results = self.detector.detect(img, landmarks=False)
        
        # Check if results contain faces
        if results is not None and len(results) > 0:
            boxes = results[0]  # Extract bounding boxes from the results
            faces = []
            for box in boxes:
                x1, y1, x2, y2 = [int(num) for num in box]
                face = img[y1:y2, x1:x2]
                face_arr = cv.resize(face, self.target_size)
                faces.append(face_arr)
            return faces

        return None
    
    def process_image(self, filename):
        faces = self.extract_face(filename)
        if faces is not None:
            # Assuming that you may want to get embeddings or process the faces further
            return faces
        else:
            print("No face detected.")
            return None

# Load the pretrained FaceNet model
# embedder = InceptionResnetV1(pretrained='vggface2').eval()
detector = MTCNN(keep_all=False)  # Initialize MTCNN for face detection
model = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(face_img):
    detector = MTCNN(keep_all=False)  # Initialize MTCNN for face detection
    model = InceptionResnetV1(pretrained='vggface2').eval()
    face_img = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
    face_img = torch.tensor(face_img).float()  # Convert to PyTorch tensor
    face_img = face_img.permute(2, 0, 1)  # Change dimensions to (C, H, W)
    face_img = face_img.unsqueeze(0)  # Add batch dimension
    face_img = (face_img - 127.5) / 128.0  # Normalize image

    with torch.no_grad():
        embedding = model(face_img)  # Get the embedding
    return embedding.squeeze().numpy()  # Convert to NumPy array

def update_npz(npz_file, new_embeddings, new_labels):
    # Check if the file exists and is not empty
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        existing_embeddings = data['embeddings'] if 'embeddings' in data else np.empty((0, new_embeddings.shape[1]))
        existing_labels = data['labels'] if 'labels' in data else np.empty((0,), dtype=object)
    except (EOFError, ValueError, KeyError) as e:
        # Handle the case where the file is empty or missing expected keys
        print(f"Error loading data from {npz_file}: {e}")
        existing_embeddings = np.empty((0, new_embeddings.shape[1]))
        existing_labels = np.empty((0,), dtype=object)
    
    # Combine existing data with new data
    updated_embeddings = np.concatenate((existing_embeddings, new_embeddings), axis=0)
    updated_labels = np.concatenate((existing_labels, new_labels), axis=0)
    
    # Save the updated data
    np.savez(npz_file, embeddings=updated_embeddings, labels=updated_labels, allow_pickle = True)

def retrain_model(npz_file, model_file):
    # Load data from .npz file
    data = np.load(npz_file, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']

    # Check if data is valid
    if embeddings.size == 0 or labels.size == 0:
        raise ValueError("Loaded data is empty. Ensure the .npz file contains embeddings and labels.")

    # Encode labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Train the SVM model
    model = SVC(kernel='linear', probability=True)  # You can adjust the kernel and parameters
    model.fit(embeddings, encoded_labels)

    # Save the trained model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model retrained and saved to {model_file}")

def recognise(pkl_file, np_file, embeddings):
    # Load the embeddings and labels
    faces_embeddings = np.load(np_file, allow_pickle=True)
    X = faces_embeddings['embeddings']  # Assuming embeddings are stored under 'arr_0'
    Y = faces_embeddings['labels']  # Assuming labels are stored under 'arr_1'

    # Load the trained model
    with open(pkl_file, 'rb') as f:
        model = pickle.load(f)

    # Fit the label encoder on existing labels
    encoder = LabelEncoder()
    encoder.fit(Y)

    # Predict the class for each embedding
    predictions = model.predict(embeddings)
    
    # Convert predictions to class names
    final_names = encoder.inverse_transform(predictions)[0]

    print(f"Predicted labels: {predictions}")
    print(f"Final names: {final_names}")

    return final_names