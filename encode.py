import os
import face_recognition

def encode_images_in_folder(image_folder):
    # Initialize an empty dictionary to store ID to username mappings
    id_to_username = {}
    # Initialize an empty list to store face encodings
    encoded_faces = []

    # Iterate through all files in the image folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Extract ID and username from filename
            user_id, username = filename.split("_")
            username = username.split(".")[0]  # Remove file extension

            # Load the image
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)

            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)
            if len(face_encodings) > 0:
                # Store the encoding and ID-username mapping
                encoded_faces.append(face_encodings[0])
                id_to_username[user_id] = username
            else:
                print(f"No face found in {filename}")

    return encoded_faces, id_to_username

# Example usage
image_folder = 'captured_images'
encoded_faces, id_to_username = encode_images_in_folder(image_folder)

# Print encoded faces and ID-username mapping
print("Encoded Faces:")
print(encoded_faces)
print("ID to Username Mapping:")
print(id_to_username)
