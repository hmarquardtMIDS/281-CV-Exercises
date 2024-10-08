import cv2
import numpy as np

def detect_motion(video_path):
    """
    Detect motion in a video and compute optical flow for each frame.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        None: Displays video with flow field visualization
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
    # Convert to grayscale
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Create random colors for flow visualization
    # color = np.random.randint(0, 255, (100, 3))
    color = np.random.randint(0, 255, (100, 3))
    
    # Find initial corner points
    feature_params = dict(
        maxCorners=20,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    p0 = cv2.goodFeaturesToTrack(
        old_gray, 
        mask=None, 
        **feature_params
        # maxCorners=100,
        # qualityLevel=0.3,
        # minDistance=7,
        # blockSize=7,
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, 
            frame_gray, 
            p0, 
            None, 
            **lk_params
        )
        
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # Draw the flow lines
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                               color[i % 100].tolist(), 2)
                
                # Draw the current points
                frame = cv2.circle(frame, (int(a), int(b)), 5, 
                                 color[i % 100].tolist(), -1)
            
            # Combine the frame with the flow field
            img = cv2.add(frame, mask)
            
            # Display the result
            cv2.imshow('Optical Flow', img)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Update the previous frame and points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = "/Users/Henry/Desktop/github/281-CV-Exercises/week_7_motion/golf_swing.mp4"
    detect_motion(video_path)