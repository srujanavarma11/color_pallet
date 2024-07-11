# import cv2
# import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt

# # Initialize mediapipe FaceMesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# def visualize_landmarks(image_path):
#     # Load the image
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (640, 480))  # Resize for faster processing
#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Perform face detection and landmark extraction
#     results = face_mesh.process(rgb_img)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             for idx, landmark in enumerate(face_landmarks.landmark):
#                 x = int(landmark.x * img.shape[1])
#                 y = int(landmark.y * img.shape[0])
#                 cv2.circle(rgb_img, (x, y), 1, (0, 255, 0), -1)
#                 cv2.putText(rgb_img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

#     else:
#         print("No face detected")

#     # Display the image with landmarks
#     plt.imshow(rgb_img)
#     plt.title("Image with Landmarks")
#     plt.axis('off')  # Turn off axis
#     plt.show()

# # Replace 'path_to_your_image.jpg' with the uploaded image path
# image_path = 'C:/Users/sruja/Downloads/Color_Palette_Analyzer-main/Color_Palette_Analyzer-main/image1.png'
# visualize_landmarks(image_path)



# import cv2
# import numpy as np
# import mediapipe as mp

# # Initialize mediapipe FaceMesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# def extract_hexcodes(image_path):
#     # Load the image
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (640, 480))  # Resize for faster processing
#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Perform face detection and landmark extraction
#     results = face_mesh.process(rgb_img)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Define regions of interest (ROI) for hair, skin, and lips
#             #78, 95, 88, 178, 152, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378
#             hair_points = [  284, 251, 389, 356]  # Example: upper forehead area
#             cheek_points = [31,32,33]  # Example: cheek area
#             lips_points = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 287]  # Example: lips area

#             # Extract colors from each region based on the specified criteria
#             hair_color = extract_darkest_color_from_bbox(rgb_img, face_landmarks, hair_points)
#             cheek_color = extract_average_color_from_bbox(rgb_img, face_landmarks, cheek_points)
#             lips_color = extract_average_color_from_bbox(rgb_img, face_landmarks, lips_points)

#             # Print hex codes
#             print("Hex codes -")
#             print("Hair color:", hair_color)
#             print("Cheek color:", cheek_color)
#             print("Lips color:", lips_color)

#             # Visualize the bounding boxes for each region
#             visualize_bounding_box(rgb_img, face_landmarks, hair_points, (255, 0, 0), "Hair")  # Red for hair
#             visualize_bounding_box(rgb_img, face_landmarks, cheek_points, (0, 255, 0), "Cheek")  # Green for cheek
#             visualize_bounding_box(rgb_img, face_landmarks, lips_points, (0, 0, 255), "Lips")  # Blue for lips

#             # Convert RGB image back to BGR for OpenCV display
#             bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
#             cv2.imshow('Image with Bounding Boxes', bgr_img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#     else:
#         print("No face detected")

# def extract_darkest_color_from_bbox(img, landmarks, points):
#     # Get bounding box coordinates
#     x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, img)

#     # Extract the region of interest (ROI) from the image
#     roi = img[y_min:y_max, x_min:x_max]

#     # Find the darkest color (minimum sum of RGB values) in the ROI
#     darkest_color = min(roi.reshape(-1, 3), key=lambda color: np.sum(color))

#     # Convert the darkest color to hex format
#     hex_color = rgb_to_hex(darkest_color)
#     return hex_color

# def extract_average_color_from_bbox(img, landmarks, points):
#     # Get bounding box coordinates
#     x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, img)

#     # Extract the region of interest (ROI) from the image
#     roi = img[y_min:y_max, x_min:x_max]

#     # Calculate the average color in the ROI
#     average_color = np.mean(roi.reshape(-1, 3), axis=0)

#     # Convert the average color to hex format
#     hex_color = rgb_to_hex(average_color.astype(int))
#     return hex_color

# def get_bbox_coordinates(landmarks, points, img):
#     x_min = y_min = float('inf')
#     x_max = y_max = float('-inf')

#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         if x < x_min:
#             x_min = x
#         if y < y_min:
#             y_min = y
#         if x > x_max:
#             x_max = x
#         if y > y_max:
#             y_max = y

#     return x_min, y_min, x_max, y_max

# def visualize_bounding_box(img, landmarks, points, color, label):
#     x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, img)
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
#     cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# def rgb_to_hex(rgb):
#     # Convert RGB to hex format
#     return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# # Replace 'path_to_your_image.jpg' with your actual image path
# image_path = 'C:/Users/sruja/Downloads/Color_Palette_Analyzer-main/Color_Palette_Analyzer-main/image2.jpg'
# extract_hexcodes(image_path)


import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def extract_hexcodes(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 480))  # Resize for faster processing
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform face detection and landmark extraction
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define regions of interest (ROI) for hair, skin, and lips
            hair_points = [284, 251, 389, 356]  # Example: upper forehead area
            cheek_points = [31, 32, 33]  # Example: cheek area
            lips_points = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 287]  # Example: lips area

            # Extract colors from each region based on the specified criteria
            hair_color = extract_darkest_color_from_bbox(rgb_img, face_landmarks, hair_points)
            cheek_color = extract_average_color_from_bbox(rgb_img, face_landmarks, cheek_points)
            lips_color = extract_average_color_from_bbox(rgb_img, face_landmarks, lips_points)

            # Print hex codes
            print("Hex codes -")
            print("Hair color:", hair_color)
            print("Cheek color:", cheek_color)
            print("Lips color:", lips_color)

            # Visualize the bounding boxes for each region
            visualize_bounding_box(rgb_img, face_landmarks, hair_points, (255, 0, 0), "Hair")  # Red for hair
            visualize_bounding_box(rgb_img, face_landmarks, cheek_points, (0, 255, 0), "Cheek")  # Green for cheek
            visualize_bounding_box(rgb_img, face_landmarks, lips_points, (0, 0, 255), "Lips")  # Blue for lips

            # Convert RGB image back to BGR for OpenCV display
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Image with Bounding Boxes', bgr_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No face detected")

def extract_darkest_color_from_bbox(img, landmarks, points):
    # Get bounding box coordinates
    x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, img)

    # Extract the region of interest (ROI) from the image
    roi = img[y_min:y_max, x_min:x_max]

    # Find the darkest color (minimum sum of RGB values) in the ROI
    darkest_color = roi.reshape(-1, 3)[np.argmin(np.sum(roi.reshape(-1, 3), axis=1))]

    # Convert the darkest color to hex format
    hex_color = rgb_to_hex(darkest_color)
    return hex_color

def extract_average_color_from_bbox(img, landmarks, points):
    # Get bounding box coordinates
    x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, img)

    # Extract the region of interest (ROI) from the image
    roi = img[y_min:y_max, x_min:x_max]

    # Calculate the average color in the ROI
    average_color = np.mean(roi.reshape(-1, 3), axis=0)

    # Convert the average color to hex format
    hex_color = rgb_to_hex(average_color.astype(int))
    return hex_color

def get_bbox_coordinates(landmarks, points, img):
    x_min = y_min = float('inf')
    x_max = y_max = float('-inf')

    for idx in points:
        x = int(landmarks.landmark[idx].x * img.shape[1])
        y = int(landmarks.landmark[idx].y * img.shape[0])
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y

    return x_min, y_min, x_max, y_max

def visualize_bounding_box(img, landmarks, points, color, label):
    x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, img)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def rgb_to_hex(rgb):
    # Convert RGB to hex format
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# Replace 'path_to_your_image.jpg' with your actual image path
image_path = 'C:/Users/sruja/Downloads/Color_Palette_Analyzer-main/Color_Palette_Analyzer-main/image4.jpg'
extract_hexcodes(image_path)

