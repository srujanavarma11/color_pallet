import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Define color palette ranges based on hexcodes
color_palettes = {
    'Soft Summer': {
        'Hair': [(0xa0522d, 0xbcafa7), (0xd1b48c, 0xe6ccb2), (0xc0c0c0, 0xc0c0c0)],
        'Lips': [(0xffb6c1, 0xe6e6fa), (0x4b0082, 0x8a2be2)],
        'Skin': [(0xc0c0c0, 0xe0e0e0)]
    },
    'Deep Winter': {
        'Hair': [(0x321414, 0x4b0082), (0x808080, 0xc0c0c0), (0x000080, 0x000080)],
        'Lips': [(0x800000, 0x4b0082), (0x8a2be2, 0x800080)],
        'Skin': [(0x321414, 0x808080)]
    },
    'Light Spring': {
        'Hair': [(0xe6b800, 0xffd700), (0xc68e17, 0xd2b48c), (0xffcc99, 0xffcc99)],
        'Lips': [(0xffc0cb, 0xff69b4), (0xff7f50, 0xff4500)],
        'Skin': [(0xffdab9, 0xffefd5)]
    },
    'Clear Spring': {
        'Hair': [(0xffd700, 0xf0e68c), (0xd2b48c, 0xcd853f), (0xff6347, 0xff7f50)],
        'Lips': [(0xff6347, 0xff4500), (0xff69b4, 0xff1493)],
        'Skin': [(0xffefd5, 0xffdab9)]
    },
    'Soft Autumn': {
        'Hair': [(0x8b4513, 0x8b0000), (0xa52a2a, 0xcd5c5c), (0xd2b48c, 0xdeb887)],
        'Lips': [(0xd2691e, 0x8b4513), (0xa52a2a, 0x800000)],
        'Skin': [(0xdeb887, 0xd2b48c)]
    },
    'Deep Autumn': {
        'Hair': [(0x2f4f4f, 0x4b0082), (0x800000, 0xa52a2a), (0xcd5c5c, 0x8b0000)],
        'Lips': [(0x800000, 0x4b0082), (0x8a2be2, 0x800080)],
        'Skin': [(0x2f4f4f, 0xcd5c5c)]
    },
    'Light Summer': {
        'Hair': [(0xe0ffff, 0xb0c4de), (0xb0c4de, 0xc0c0c0), (0xc0c0c0, 0xc0c0c0)],
        'Lips': [(0xffb6c1, 0xe6e6fa), (0x4b0082, 0x8a2be2)],
        'Skin': [(0xe0ffff, 0xc0c0c0)]
    },
    'Clear Winter': {
        'Hair': [(0x000000, 0x000000), (0xe0ffff, 0xc0c0c0), (0x808080, 0xc0c0c0)],
        'Lips': [(0xff0000, 0xff1493), (0x800080, 0x4b0082)],
        'Skin': [(0x808080, 0xc0c0c0)]
    }
}

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

            # Convert hex codes to integers for comparison
            hair_color_int = int(hair_color[1:], 16)
            lips_color_int = int(lips_color[1:], 16)
            cheek_color_int = int(cheek_color[1:], 16)

            # Check which color palette the extracted colors belong to
            for palette_name, palette_ranges in color_palettes.items():
                hair_range = palette_ranges['Hair']
                lips_range = palette_ranges['Lips']
                skin_range = palette_ranges['Skin']

                hair_match = all(hair_range[i][0] <= hair_color_int <= hair_range[i][1] for i in range(len(hair_range)))
                lips_match = all(lips_range[i][0] <= lips_color_int <= lips_range[i][1] for i in range(len(lips_range)))
                skin_match = all(skin_range[i][0] <= cheek_color_int <= skin_range[i][1] for i in range(len(skin_range)))

                if hair_match and lips_match and skin_match:
                    print(f"Detected color palette: {palette_name}")

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
image_path = 'C:/Users/sruja/Downloads/Color_Palette_Analyzer-main/Color_Palette_Analyzer-main/image1.png'
extract_hexcodes(image_path)


