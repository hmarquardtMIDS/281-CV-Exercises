import cv2
import numpy as np

def video_gradient(input_video_path, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height), False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute the gradient using Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute the magnitude of the gradient
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Normalize the gradient to the range [0, 255]
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        img = np.uint8(gradient_magnitude)
        
        # Write the frame to the output video
        # out.write(gradient_magnitude)

        # Display the result
        cv2.imshow('Gradient', img)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
# input_video_path = './week_7_motion/golf_swing.mp4'
input_video_path = './week_7_motion/cars.mp4'
output_video_path = './week_7_motion/output_gradient_video.mp4'
video_gradient(input_video_path, output_video_path)