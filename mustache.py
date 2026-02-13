import cv2
import itertools
import numpy as np
import mediapipe as mp
import os

# Initialize Mediapipe components
mp_face_mesh = mp.solutions.face_mesh
face_mesh_videos = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks for the lips region
LIPS_INDEXES = list(itertools.chain(*mp_face_mesh.FACEMESH_LIPS))

def overlay(image, filter_img, face_landmarks):
    """Overlays the filter image onto the frame using proper alpha blending and boundary clipping."""
    
    image_height, image_width, _ = image.shape
    
    # Check for 4-channel (BGRA) image
    if filter_img.shape[2] < 4:
        return image
        
    # Get landmarks around the mouth for sizing
    landmarks = np.array([[int(face_landmarks.landmark[i].x * image_width),
                           int(face_landmarks.landmark[i].y * image_height)] for i in LIPS_INDEXES])
    
    x, y, w_mouth, h_mouth = cv2.boundingRect(landmarks)
    
    # --- FIX: Reduced multiplier from 2.0 to 1.3 ---
    target_width = int(w_mouth * 1.3)
    
    filter_h_w_ratio = filter_img.shape[0] / filter_img.shape[1]
    target_height = int(target_width * filter_h_w_ratio)
    
    if target_width <= 0 or target_height <= 0: return image

    # Center the mustache slightly below the nose
    nose_tip = face_landmarks.landmark[1]
    upper_lip = face_landmarks.landmark[13]
    
    center_x = int(nose_tip.x * image_width)
    center_y = int((nose_tip.y * 0.5 + upper_lip.y * 0.5) * image_height)
    
    # Define the target bounding box
    x1_target = center_x - target_width // 2
    y1_target = center_y - target_height // 2
    
    # --- Safe Boundary Clipping ---
    x1_clip = max(x1_target, 0)
    y1_clip = max(y1_target, 0)
    x2_clip = min(x1_target + target_width, image_width)
    y2_clip = min(y1_target + target_height, image_height)
    
    if x1_clip >= x2_clip or y1_clip >= y2_clip: return image

    # Resize the mustache filter
    resized_filter = cv2.resize(filter_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Calculate offsets for clipping
    x1_offset = x1_clip - x1_target
    y1_offset = y1_clip - y1_target
    x2_offset = x1_offset + (x2_clip - x1_clip)
    y2_offset = y1_offset + (y2_clip - y1_clip)
    
    overlay_clipped = resized_filter[y1_offset:y2_offset, x1_offset:x2_offset]
    roi = image[y1_clip:y2_clip, x1_clip:x2_clip].copy()

    # --- Alpha Blending ---
    overlay_bgr = overlay_clipped[:, :, :3].astype(np.float32)
    alpha = overlay_clipped[:, :, 3].astype(np.float32) / 255.0
    
    alpha_mask = cv2.merge([alpha, alpha, alpha])
    
    blended = (overlay_bgr * alpha_mask) + (roi.astype(np.float32) * (1.0 - alpha_mask))
    
    image[y1_clip:y2_clip, x1_clip:x2_clip] = blended.astype(np.uint8)
        
    return image


def apply_mustache(frame):
    """Applies the mustache filter to a single frame."""
    mustache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media", "mustache1.png")
    mustache = cv2.imread(mustache_path, cv2.IMREAD_UNCHANGED)

    if mustache is None:
        # Fail silently by returning the frame so the app doesn't crash
        return frame
    
    # If the image loaded without alpha channel (BGR), convert it to BGRA
    if mustache.shape[2] == 3:
         mustache = cv2.cvtColor(mustache, cv2.COLOR_BGR2BGRA)

    face_mesh_results = face_mesh_videos.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    filtered_frame = frame.copy()

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            filtered_frame = overlay(filtered_frame, mustache, face_landmarks)

    return filtered_frame