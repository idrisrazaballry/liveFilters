import cv2

def apply_oilpaint(frame):
    """
    Applies the oil paint filter to a single input frame.
    """
    try:
        # Apply oil paint effect 
        oil_painted_frame = cv2.xphoto.oilPainting(frame, 7, 1)

        # Check if output is grayscale and convert back to BGR
        if len(oil_painted_frame.shape) == 2:
            oil_painted_frame = cv2.cvtColor(oil_painted_frame, cv2.COLOR_GRAY2BGR)
            
        return oil_painted_frame
    except AttributeError:
        # Failsafe if opencv-contrib-python is not installed (xphoto is missing)
        print("Error: cv2.xphoto.oilPainting is not available. Returning original frame.")
        return frame
    except Exception as e:
        print(f"Oil Paint Error: {e}")
        return frame