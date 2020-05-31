import face_recognition
import cv2
import numpy as np

# Load the known images
image_of_person_11 = face_recognition.load_image_file("rashmi1.jpg")
image_of_person_12 = face_recognition.load_image_file("rashmi2.jpg")
image_of_person_21 = face_recognition.load_image_file("chanch1.jpg")
image_of_person_22 = face_recognition.load_image_file("chanch2.jpg")
image_of_person_31 = face_recognition.load_image_file("harmeet1.jpg")
image_of_person_32 = face_recognition.load_image_file("harmeet2.jpg")


# Get the face encoding of each person. This can fail if no one is found in the photo.
person_11_face_encoding = face_recognition.face_encodings(image_of_person_11)[0]
person_12_face_encoding = face_recognition.face_encodings(image_of_person_12)[0]
person_21_face_encoding = face_recognition.face_encodings(image_of_person_21)[0]
person_22_face_encoding = face_recognition.face_encodings(image_of_person_22)[0]
person_31_face_encoding = face_recognition.face_encodings(image_of_person_31)[0]
person_32_face_encoding = face_recognition.face_encodings(image_of_person_32)[0]

# Create a list of all known face encodings
known_face_encodings = [
    person_11_face_encoding,  person_12_face_encoding,
    person_21_face_encoding,   person_22_face_encoding,
    person_31_face_encoding, person_32_face_encoding
]

#Known face names
known_face_names = [
    "Rashmi", "Rashmi",
    "Chanchal",  "Chanchal",
    "Harmeet", "Harmeet"
]


# Load the image we want to check
unknown_image1 = face_recognition.load_image_file("group3.jpg")
#unknown_image1 = face_recognition.load_image_file("group4.jpg")

# Resize frame of video to 1/4 size for faster face recognition processing
#small_frame = cv2.resize(unknown_image1, (0, 0), fx=0.25, fy=0.25)
 # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#rgb_small_frame = small_frame[:, :, ::-1]

# Get face encodings for any people in the picture
un_face_locations = face_recognition.face_locations(unknown_image1, number_of_times_to_upsample=2)
unknown_face_encodings = face_recognition.face_encodings(unknown_image1, un_face_locations)

number_of_faces = len(un_face_locations)
print("I found {} face(s) in this photograph.".format(number_of_faces))

# Load the image into a Python Image Library object so that we can draw on top of it and display it
#pil_image = PIL.Image.fromarray(unknown_image1)

# Initialize some variables
face_names = []
# There might be more than one person in the photo, so we need to loop over each face we found
for unknown_face_encoding in unknown_face_encodings:
    # Test if this unknown face encoding matches any of the three people we know
    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.5)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
    best_match_index = np.argmin(face_distances)
    if results[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)
    print(f"Found {name} in the photo!")



# Display the results
for (top, right, bottom, left), name in zip(un_face_locations, face_names):
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #top *= 4
    #right *= 4
    #bottom *= 4
    #left *= 4
    # Draw a box around the face
    cv2.rectangle(unknown_image1, (left, top), (right, bottom), (0, 255, 0), 10)
    # Draw a label with a name below the face
   # cv2.rectangle(unknown_image1, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(unknown_image1, name, (left, y), font, 1.0, (0, 255, 0), 2)

# Display the resulting image
small_frame_new = cv2.resize(unknown_image1, (0, 0), fx=0.50, fy=0.50)
#rgb_small_frame_new = small_frame_new[:, :, ::-1]
small_frame_new = small_frame_new[:, :, ::-1]
cv2.imshow('Result', small_frame_new)
cv2.waitKey(0)
