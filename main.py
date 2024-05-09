import cv2
import face_recognition
import numpy as np
import pandas as pd
from encode import encoded_faces, id_to_username
from datetime import datetime

face_locations = []
face_encodings = []
face_names = []

# Real time video for face detection
video_capture = cv2.VideoCapture(0)

# Initialize DataFrame to store data
data = []
curr_names = []

while True:
    # Capture frames
    ret, frame = video_capture.read()

    # Resize frame to 1/4th of its original 
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face from video and enocde it
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # List of faces in the video frame
    face_names = []
    for face_encoding in face_encodings:

        # Matching video face to known faces in encode variable
        matches = face_recognition.compare_faces(encoded_faces, face_encoding)
        name = "Unknown"

        # Checking for True in list matches and its index
        if True in matches:
            # Find the index of the first True in match
            first_match_index = matches.index(True)
            # Get the name of the face using the index
            user_id = list(id_to_username.keys())[first_match_index]
            name = id_to_username[user_id]

        face_names.append(name)

    # Store data for known faces
    for name in face_names:
        if name != "Unknown" and name not in curr_names:
            user_id = [key for key, value in id_to_username.items() if value == name][0]
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data.append([name, user_id, time])
            curr_names.append(name)
            
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown faces
        else:
            color = (0, 255, 0)  # Green for known faces

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom-35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        print(left, top, right, bottom)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

# Create a DataFrame from the collected data
df = pd.DataFrame(data, columns=['Name', 'ID', 'Time'])

# Write DataFrame to Excel file
excel_file = 'face_recognition_data.xlsx'
df.to_excel(excel_file, index=False)
print(f"Data saved to {excel_file}")
