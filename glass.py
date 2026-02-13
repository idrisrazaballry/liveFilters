import cv2
import itertools
import numpy as np
import mediapipe as mp

# --- MEDIAPIPE SETUP (CRITICAL) ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Use face_mesh_videos for frame-by-frame processing
face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2,
                                         min_detection_confidence=0.5, min_tracking_confidence=0.3)

mp_drawing_styles = mp.solutions.drawing_styles


# --- HELPER FUNCTIONS (CRITICAL) ---

def detectFacialLandmarks(image, face_mesh, display=False):
    """Detect facial landmarks using MediaPipe FaceMesh."""
    # Convert BGR to RGB for MediaPipe processing
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    output_image = image.copy()
    if not display:
        return output_image, results


def getSize(image, face_landmarks, INDEXES):
    """
    Calculate bounding box size and landmarks array for specific facial regions.
    INDEXES: list of landmark indices to calculate around.
    """
    image_height, image_width, _ = image.shape
    INDEXES_LIST = list(itertools.chain(*INDEXES))
    landmarks = []
    for INDEX in INDEXES_LIST:
        landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),
                          int(face_landmarks.landmark[INDEX].y * image_height)])

    _, _, width, height = cv2.boundingRect(np.array(landmarks))
    landmarks = np.array(landmarks)
    return width, height, landmarks


def overlay(image, filter_img, face_landmarks, face_part, INDEXES, display=False):
    """
    Overlay the filter_img on specified face part defined by face_landmarks and landmark INDEXES.
    """
    annotated_image = image.copy()
    try:
        filter_img_height, filter_img_width, _ = filter_img.shape
        _, face_part_height, landmarks = getSize(image, face_landmarks, INDEXES)

        # Scale filter size for proper fitting
        required_height = int(face_part_height * 13)
        resized_filter_img = cv2.resize(filter_img, (int(filter_img_width * (required_height / filter_img_height)), required_height))

        filter_img_height, filter_img_width, _ = resized_filter_img.shape

        # Create mask from the asset's grayscale and threshold for transparency
        gray_filter = cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY)
        _, filter_img_mask = cv2.threshold(gray_filter, 0, 255, cv2.THRESH_BINARY_INV)

        center = landmarks.mean(axis=0).astype("int")

        # Calculate overlay position based on face part
        if face_part == 'MOUTH':
            location = (int(center[0] - filter_img_width / 3), int(center[1]))
        else:
            # Default placement for eye/face filters (like glasses)
            location = (int(center[0] - filter_img_width / 1.3), int(center[1] - filter_img_height / 2))

        end_y = location[1] + filter_img_height
        end_x = location[0] + filter_img_width

        # Check if overlay location is fully in image boundaries
        if location[1] >= 0 and location[0] >= 0 and end_y <= image.shape[0] and end_x <= image.shape[1]:
            ROI = image[location[1]: end_y, location[0]: end_x]

            # Masking and combining filter image with ROI
            resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)
            resultant_image = cv2.add(resultant_image, resized_filter_img)

            # Place combined image back into frame
            annotated_image[location[1]: end_y, location[0]: end_x] = resultant_image

    except Exception as e:
        # You can uncomment this for debugging overlay issues
        # print(f"Overlay error: {e}")
        pass

    return annotated_image


# --- MAIN FUNCTION (Takes one argument) ---

def apply_glass(frame):
    """
    Applies the glasses filter to a single input frame.
    """
    glasses = cv2.imread('media/glasses3.png')
    if glasses is None:
        return frame

    _, face_mesh_results = detectFacialLandmarks(frame, face_mesh_videos, display=False)

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            frame = overlay(frame, glasses, face_landmarks,
                            'LEFT EYE', mp_face_mesh.FACEMESH_LEFT_EYE, display=False)

    return frame
