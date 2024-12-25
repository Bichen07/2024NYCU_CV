import cv2 as cv
import matplotlib.pyplot as plt

def imshow(image, title=""):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Read images
src = cv.imread("input_data/2target_object.png")  # Source image
dst = cv.imread("input_data/2background.jpg")    # Background image
mask = cv.imread("input_data/2target_mask.png", 0)  # Binary mask (grayscale)

# Define the position where the source object is placed in the background
center = (300, 600)  # Example: (x, y) coordinates in the destination image

# Perform seamless cloning
output = cv.seamlessClone(src, dst, mask, center, cv.NORMAL_CLONE)

# Display the result
imshow(output, "Seamless Clone Output")
