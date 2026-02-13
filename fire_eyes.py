import os, cv2, mediapipe as mp, numpy as np, math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "media")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True, # Essential for pupil landmarks (468, 473)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def apply_fire_eyes(frame):
    """Applies the fire eyes filter, compatible with main.py's frame processing loop."""
    try:
        left_path = os.path.join(MEDIA_DIR, "fire_left.png")
        right_path = os.path.join(MEDIA_DIR, "fire_right.png")
        
        # Load the images
        left_fire = cv2.imread(left_path, cv2.IMREAD_UNCHANGED)
        right_fire = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)
        
        if left_fire is None or right_fire is None:
            return frame
        
        # Ensure images have 4 channels (BGRA) for transparency
        if left_fire.shape[2] < 4:
            left_fire = cv2.cvtColor(left_fire, cv2.COLOR_BGR2BGRA)
        
        if right_fire.shape[2] < 4:
            right_fire = cv2.cvtColor(right_fire, cv2.COLOR_BGR2BGRA)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return frame

        h, w = frame.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        
        # --- FIXED LOGIC: Use Pupils and Eye Width for sizing/centering ---
        
        # Pupil landmarks: 473 (Left Eye), 468 (Right Eye) for perfect center
        left_pupil = lm[473] 
        right_pupil = lm[468]

        # Calculate pixel distance of one eye (from inner corner 33 to outer corner 133)
        eye_width_px = math.hypot((lm[33].x - lm[133].x)*w, (lm[33].y - lm[133].y)*h)
        
        # Target size: 3.5 times the eye width to cover the eye and surrounding area
        target_size = int(eye_width_px * 3.5) 
        
        # Determine height based on aspect ratio
        left_h_w_ratio = left_fire.shape[0] / left_fire.shape[1]
        right_h_w_ratio = right_fire.shape[0] / right_fire.shape[1]
        
        target_left_h = int(target_size * left_h_w_ratio)
        target_right_h = int(target_size * right_h_w_ratio)

        # Iterate over both eyes
        for img, pupil, target_h in [
            (left_fire, left_pupil, target_left_h), 
            (right_fire, right_pupil, target_right_h)
        ]:
            if target_size <= 0 or target_h <= 0: continue

            cx, cy = int(pupil.x * w), int(pupil.y * h)
            
            # Define the target bounding box centered on the pupil
            x1_target = cx - target_size // 2
            y1_target = cy - target_h // 2
            
            # --- Safe Boundary Clipping ---
            x1_clip = max(x1_target, 0)
            y1_clip = max(y1_target, 0)
            x2_clip = min(x1_target + target_size, w)
            y2_clip = min(y1_target + target_h, h)

            if x1_clip >= x2_clip or y1_clip >= y2_clip: continue

            # Resize the filter image
            resized_fire = cv2.resize(img, (target_size, target_h), interpolation=cv2.INTER_AREA)

            # Calculate offsets for clipping
            x1_offset = x1_clip - x1_target
            y1_offset = y1_clip - y1_target
            x2_offset = x1_offset + (x2_clip - x1_clip)
            y2_offset = y1_offset + (y2_clip - y1_clip)
            
            overlay_clipped = resized_fire[y1_offset:y2_offset, x1_offset:x2_offset]
            roi = frame[y1_clip:y2_clip, x1_clip:x2_clip].copy()

            # --- Alpha Blending ---
            overlay_bgr = overlay_clipped[:, :, :3].astype(np.float32)
            alpha = overlay_clipped[:, :, 3].astype(np.float32) / 255.0
            
            alpha_mask = cv2.merge([alpha, alpha, alpha])
            
            blended = (overlay_bgr * alpha_mask) + (roi.astype(np.float32) * (1.0 - alpha_mask))
            
            frame[y1_clip:y2_clip, x1_clip:x2_clip] = blended.astype(np.uint8)
            
        return frame
    except Exception as e:
        # print(" Fire Eyes Filter Runtime Error:", e) # Uncomment if debugging is needed
        return frame