
import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from datetime import datetime
import os
import time
import random
import pandas as pd
import sqlite3
import shutil
import uuid





# SQLite database file
db_file = 'faces.db'






# Create the SQLite database and table
def create_db():
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # Create a table to store person's name and image path
    c.execute('''CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    image_path TEXT)''')
    conn.commit()
    conn.close()

# Function to save a person's image to the SQLite database
def save_face_image(image_path, name):
    # Copy the image to a "database" folder (optional, to store all images)
    db_image_folder = 'db_images'
    if not os.path.exists(db_image_folder):
        os.makedirs(db_image_folder)
    
    # Create a unique image path for saving
    image_filename = f"{name}.jpg"
    image_save_path = os.path.join(db_image_folder, image_filename)
    
    # Copy image to the database folder
    shutil.copy(image_path, image_save_path)
    
    # Store the image path and name in the SQLite database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('INSERT INTO faces (name, image_path) VALUES (?, ?)', (name, image_save_path))
    conn.commit()
    conn.close()

    print(f"Image for {name} saved to the database.")




# Function to recognize a face from an input image using DeepFace and compare with stored images
def recognize_face(image_path):
    # Use DeepFace to compare the input image with the stored images in the database
    result = pd.DataFrame()
    result = DeepFace.find(image_path, db_path='db_images', enforce_detection=False, threshold= 0.6)
    # Combine the list of DataFrames into a single DataFrame
    combined_df = pd.concat(result, ignore_index=True)
    
    st.text(len(combined_df))
    st.text(combined_df)
    return combined_df




def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    return faces


def save_frames():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not access the webcam.")
        return

    # Create output directory
    output_dir = r"D:\Education\courses\MachineLearning\debi\05_projects\facedetection"
    os.makedirs(output_dir, exist_ok=True)

    st.info("Capturing frames. Uncheck 'Start Camera' to stop.")
    frame_display = st.empty()

    try:
        while st.session_state["run_camera"]:
            # Capture a frame
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame. Exiting.")
                break

            # Apply face detection
            faces = detect_bounding_box(frame)

            # If faces are detected, crop the face region with an offset
            if len(faces) != 0:
                (x, y, w, h) = faces[0]  # Assuming the first face is the target one
                
                # Add a small offset to the bounding box (e.g., 20 pixels)
                offset = 20
                x_offset = max(x - offset, 0)
                y_offset = max(y - offset, 0)
                w_offset = min(x + w + offset, frame.shape[1])  # Ensure within frame width
                h_offset = min(y + h + offset, frame.shape[0])  # Ensure within frame height
                
                # Crop the face with the adjusted bounding box
                cropped_face = frame[y_offset:h_offset, x_offset:w_offset]
                
                # Save the cropped face image
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_path = os.path.join(output_dir, f"frame_{current_time}.jpg")
                cv2.imwrite(file_path, cropped_face)
                st.info(f"Face saved as: {file_path}")
                
                result = recognize_face(file_path)
                if(len(result) == 0):
                    random_id = str(uuid.uuid4())[:4]
                    save_face_image(file_path, 'UNKNOWN'+random_id)
                else:
                    st.text(result)
                
                
                os.remove(file_path)

            # Convert frame from BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display.image(frame_rgb, caption="Live Feed", use_container_width=True)

    finally:
        cap.release()
        st.success("Webcam released. Camera stopped.")


# Streamlit UI
st.title("Capture Image and Save as JPG")



if "run_camera" not in st.session_state:
    st.session_state["run_camera"] = False

run_camera = st.checkbox("Start Camera", value=st.session_state["run_camera"])
st.session_state["run_camera"] = run_camera


# Initialize the database (call this only once)
create_db()


# Paths to images
image_path_to_save = r'D:\Education\courses\MachineLearning\debi\05_projects\facedetection\frame_2025-01-24_13-09-01.jpg'  # The path of the image to save
# Save a face image to the database
save_face_image(image_path_to_save, 'Bishoy')


if run_camera:
    save_frames()
