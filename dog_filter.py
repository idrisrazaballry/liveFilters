import os, cv2, mediapipe as mp, numpy as np

# Set up paths relative to the script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "media")

# Initialize Mediapipe FaceMesh once
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def apply_dogfilter(frame):
    """
    Applies the dog filter image overlay with fine-tuned placement and robust alpha blending.
    """
    try:
        # Load the image with the alpha channel (IMREAD_UNCHANGED)
        dog_path = os.path.join(MEDIA_DIR, "dog_face_filter.png")
        dog = cv2.imread(dog_path, cv2.IMREAD_UNCHANGED)
        
        # Check for 4-channel image (BGRA)
        if dog is None or dog.shape[2] < 4:
            print(f"❌ Dog Filter: Image not found or is not a 4-channel transparent PNG.")
            return frame
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return frame

        h, w = frame.shape[:2]
        lm = res.multi_face_landmarks[0].landmark

        # --- Use specific landmarks for better anchoring ---
        # For overall width, use outer eye/temple region for stability
        left_outer_eye = lm[226] # Or similar outer landmark for width
        right_outer_eye = lm[446] # Or similar outer landmark for width
        nose_tip = lm[1]
        chin = lm[152]
        
        # Calculate target filter width based on the width between outer eye regions
        # This provides a more stable horizontal anchor
        filter_anchor_width = abs(right_outer_eye.x - left_outer_eye.x) * w
        
        # Adjust the multiplier for the overall filter width
        # The value '2.5' might need fine-tuning for your specific filter image
        target_width = int(filter_anchor_width * 2.5) 
        
        # Maintain aspect ratio of the filter image
        filter_h_w_ratio = dog.shape[0] / dog.shape[1]
        target_height = int(target_width * filter_h_w_ratio)

        # Center point: This is crucial for vertical placement.
        # Let's try anchoring the filter center slightly above the nose tip,
        # or a blend between the forehead (lm[10]) and nose tip (lm[1])
        center_x = int(nose_tip.x * w)
        # Vertical center: blend between forehead and nose tip, adjusted by a factor
        # This will lift the ears higher and position the nose correctly
        center_y = int((lm[10].y * 0.3 + nose_tip.y * 0.7) * h) - int(target_height * 0.1) 
        
        # Define the target bounding box
        x1_target = center_x - target_width // 2
        y1_target = center_y - target_height // 2
        
        # --- Safe Boundary Clipping ---
        x1_clip = max(x1_target, 0)
        y1_clip = max(y1_target, 0)
        x2_clip = min(x1_target + target_width, w)
        y2_clip = min(y1_target + target_height, h)
        
        if x1_clip >= x2_clip or y1_clip >= y2_clip:
            return frame

        # Resize the dog filter to the *original* target size
        resized_dog = cv2.resize(dog, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Get the corresponding slice of the *filter* image (for clipping)
        x1_offset = x1_clip - x1_target
        y1_offset = y1_clip - y1_target
        x2_offset = x1_offset + (x2_clip - x1_clip)
        y2_offset = y1_offset + (y2_clip - y1_clip)
        
        overlay_clipped = resized_dog[y1_offset:y2_offset, x1_offset:x2_offset]
        roi = frame[y1_clip:y2_clip, x1_clip:x2_clip].copy()

        # --- Alpha Blending ---
        overlay_bgr = overlay_clipped[:, :, :3].astype(np.float32)
        alpha = overlay_clipped[:, :, 3].astype(np.float32) / 255.0
        
        alpha_mask = cv2.merge([alpha, alpha, alpha])
        
        blended = (overlay_bgr * alpha_mask) + (roi.astype(np.float32) * (1.0 - alpha_mask))
        
        frame[y1_clip:y2_clip, x1_clip:x2_clip] = blended.astype(np.uint8)
             
        return frame
        
    except Exception as e:
        # print(f"🐶 Dog Filter Error: {e}") # Uncomment for debugging
        return frame