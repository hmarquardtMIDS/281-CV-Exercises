import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import sepfir2d
from IPython.display import clear_output
from time import sleep
from cv2 import pyrDown
import imageio
import cv2


def image_filtering_edges(im_path:str):
    """ code to compute the directional derivatives by cycling through all of the horizontal and vertical thetas.
    
    Applies edge detection filtering to an image and displays the result for different angles.
    This function reads an image file named 'disc.png', converts it to grayscale if necessary,
    and then applies separable 2D FIR filters to compute the spatial derivatives in the x and y directions.
    It then combines these derivatives at different angles (from 0 to 360 degrees in 5-degree increments)
    and displays the resulting images.
    """

    img = plt.imread(im_path)
    # Ensure the image is grayscale
    if img.ndim == 3:
        img = np.mean(img, axis=2)

    # Scale down the image if its dimensions are greater than 300
    max_dim = 300
    while img.shape[0] > max_dim or img.shape[1] > max_dim:
        img = pyrDown(img)
    
    p = [0.030320, 0.249724, 0.439911, 0.249724, 0.030320]
    d = [-0.104550, -0.292315, 0.0, 0.292315, 0.104550]

    img_x = sepfir2d(img, d, p) # spatial x derivative
    img_y = sepfir2d(img, p, d) # spatial y derivative

    for th in range (0, 361, 5):
        plt.figure(figsize=(10, 10))
        plt.imshow(np.cos(np.radians(th))*img_x + np.sin(np.radians(th))*img_y, cmap = 'gray')
        plt.title(th)
        plt.show()
        sleep(0.1)
        clear_output(wait=True)
    return None

def crop_to_square(img: np.ndarray):
    min_dim = min(img.shape[0], img.shape[1])
    start_x = (img.shape[1] - min_dim) // 2
    start_y = (img.shape[0] - min_dim) // 2
    img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    # img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return img


def save_edge_detection_gif(im_path: str, gif_path: str):
    """ Applies edge detection filtering to an image and saves the result as a GIF.
    This function reads an image file, applies the edge detection filtering at different angles,
    and saves the resulting images as frames in a GIF file.
    """
    img = plt.imread(im_path)
    # Ensure the image is grayscale
    if img.ndim == 3:
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img
    
    # Crop the image to be square
    # img_gray = crop_to_square(img_gray)

    # Scale down the image if its dimensions are greater than 300
    max_dim = 1000
    while img_gray.shape[0] > max_dim or img_gray.shape[1] > max_dim:
        img_gray = pyrDown(img_gray)
        img = pyrDown(img)
    
    p = [0.030320, 0.249724, 0.439911, 0.249724, 0.030320]
    d = [-0.104550, -0.292315, 0.0, 0.292315, 0.104550]

    img_x = sepfir2d(img_gray, d, p) # spatial x derivative
    img_y = sepfir2d(img_gray, p, d) # spatial y derivative

    frames = []
    for th in range(0, 361, 5):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
        # plt.tight_layout()

        # Edge detection subplot
        img_directional_edges = np.cos(np.radians(th)) * img_x + np.sin(np.radians(th)) * img_y
        ax1.imshow(img_directional_edges, cmap='gray')
        ax1.set_title(f'Edge Detection at {th} degrees')

        # Original image subplot with red line
        ax2.imshow(img)
        center = (img.shape[1] // 2, img.shape[0] // 2)
        length = max(img.shape[0], img.shape[1])

        # Calculate the end point of the line
        x_start = center[0] - length * np.cos(np.radians(th))
        x_end = center[0] + length * np.cos(np.radians(th))
        y_start = center[1] + length * np.sin(np.radians(th))
        y_end = center[1] - length * np.sin(np.radians(th))

        # Ensure the line does not extend beyond the edges of the image
        x_start = np.clip(x_start, 0, img.shape[1] - 1)
        x_end = np.clip(x_end, 0, img.shape[1] - 1)
        y_start = np.clip(y_start, 0, img.shape[0] - 1)
        y_end = np.clip(y_end, 0, img.shape[0] - 1)
        
        ax2.plot([x_start, x_end], [y_start, y_end], 'r-')
        ax2.set_title(f'Original Image with Red Line showing {th} degrees')
        fig.canvas.draw()

        # Convert plot to image
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(gif_path, frames, fps=10, loop=0)

# Example usage
# save_edge_detection_gif("henry.jpg", "henry_edges.gif")
# save_edge_detection_gif("grid.png", "grid_edges.gif")
# save_edge_detection_gif("disc.png", "disc_edges.gif")
# save_edge_detection_gif("bolt.jpg", "bolt_edges.gif")

# image_filtering_edges("disc.png")
image_filtering_edges("henry.jpg")


def compute_and_plot_gradient(im_path: str, resize:bool = False):
    """Computes the gradient of an image using cv2 and plots the result with matplotlib."""
    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

    # if resize:

    # Compute gradients along the x and y axis
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the gradient magnitude
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Plot the gradient magnitude
    plt.figure(figsize=(10, 10))
    plt.imshow(grad_magnitude, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    plt.show()

# Example usage
compute_and_plot_gradient("henry.jpg")
compute_and_plot_gradient("grid.png")
compute_and_plot_gradient("disc.png")
compute_and_plot_gradient("bolt.jpg")