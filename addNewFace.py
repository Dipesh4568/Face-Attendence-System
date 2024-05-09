import cv2
import face_recognition
import os

def capture_face_and_save(image_folder):
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)

    # Ask user for username and ID
    username = input("Enter your username: ")
    user_id = input("Enter your ID: ")

    while True:
        ret, frame = video_capture.read()

        # Display the captured frame
        cv2.imshow('Video', frame)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            # Save the captured frame to the specified folder with user ID and name as filename
            image_path = os.path.join(image_folder, f'{user_id}_{username}.jpg')
            cv2.imwrite(image_path, frame)
            print(f"Face captured and saved to: {image_path}")

            # Once a face is detected and saved, stop capturing
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture
    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
image_folder = 'captured_images'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

capture_face_and_save(image_folder)
