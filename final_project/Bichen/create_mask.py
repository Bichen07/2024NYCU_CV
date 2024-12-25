import argparse
import numpy as np
import cv2
from os import path

class MaskPainter():
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Could not load the image.")

        self.image_copy = self.image.copy()  # Backup for resetting
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)  # Binary mask (single channel)
        self.drawing = False
        self.erasing = False
        self.brush_size = 5  # Size of the brush
        self.window_name = "Paint Mask (Press 's' to save, 'r' to reset, 'q' to quit, 'f' to fill, 'g' to segment based on edges)"

        self.history = []  # History stack for undo functionality
        self.redo_stack = []  # Redo stack

    def _draw_mask(self, event, x, y, flags, param):
        """Mouse callback for painting or erasing the mask."""
        overlay = self.image_copy.copy()  # Base image for blending
        alpha = 0.5  # Transparency value

        if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
            self.drawing = True
            self.erasing = False
            self.history.append((self.image.copy(), self.mask.copy()))  # Save state to history
            self.redo_stack.clear()
        elif event == cv2.EVENT_RBUTTONDOWN:  # Start erasing
            self.drawing = False
            self.erasing = True
            self.history.append((self.image.copy(), self.mask.copy()))  # Save state to history
            self.redo_stack.clear()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:  # Paint while moving
                cv2.circle(self.image, (x, y), self.brush_size, (0, 255, 0), -1)
                cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
            elif self.erasing:  # Erase while moving
                cv2.circle(self.image, (x, y), self.brush_size, (0, 0, 0), -1)
                cv2.circle(self.mask, (x, y), self.brush_size, 0, -1)
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:  # Stop drawing/erasing
            self.drawing = False
            self.erasing = False

        # Blend the mask with the original image
        color_mask = np.zeros_like(self.image, dtype=np.uint8)
        color_mask[self.mask > 0] = (0, 255, 0)  # Highlight in green
        blended = cv2.addWeighted(overlay, alpha, color_mask, 1 - alpha, 0)
        self.image = blended

    def fill_mask(self):
        """Fill the inside of the drawn mask."""
        self.history.append((self.image.copy(), self.mask.copy()))  # Save state to history
        self.redo_stack.clear()
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(self.mask)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
        self.mask = filled_mask
        filled_image = self.image_copy.copy()
        filled_image[self.mask > 0] = (0, 255, 0)  # Highlight the filled region
        self.image = filled_image
        
    def segment_based_on_edges(self):
        """Segment the area around the mask based on color gradients."""
        self.history.append((self.image.copy(), self.mask.copy()))  # Save state to history
        self.redo_stack.clear()
        gray = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        gradient_threshold = 50

        filled_mask = np.zeros_like(self.mask, dtype=np.uint8)
        seeds = np.argwhere(self.mask > 0).tolist()
        visited = set()

        while len(seeds) > 0:
            x, y = seeds[0]
            seeds = seeds[1:]

            if (x, y) in visited:
                continue

            visited.add((x, y))

            filled_mask[x, y] = 255

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                if 0 <= nx < self.mask.shape[0] and 0 <= ny < self.mask.shape[1]:
                    if (nx, ny) not in visited and filled_mask[nx, ny] == 0:
                        if gradient_magnitude[nx, ny] < gradient_threshold:
                            seeds.append((nx, ny))

        edges = cv2.Canny(filled_mask, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_connected = cv2.dilate(edges, kernel, iterations=1)

        segmented_mask = cv2.bitwise_or(filled_mask, edges_connected)

        self.mask = segmented_mask
        segmented_image = self.image_copy.copy()
        segmented_image[self.mask > 0] = (0, 255, 0)  # Highlight the filled region
        self.image = segmented_image

    def undo(self):
        """Undo the last action."""
        if self.history:
            self.redo_stack.append((self.image.copy(), self.mask.copy()))
            self.image, self.mask = self.history.pop()
        else:
            print("Nothing to undo.")

    def redo(self):
        """Redo the last undone action."""
        if self.redo_stack:
            self.history.append((self.image.copy(), self.mask.copy()))
            self.image, self.mask = self.redo_stack.pop()
        else:
            print("Nothing to redo.")

    def create_mask(self):
        """Main loop for painting the mask."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._draw_mask)

        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):  # Reset the image and mask
                self.image = self.image_copy.copy()
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            elif key == ord('f'):  # Fill the inside of the mask
                self.fill_mask()
            elif key == ord('g'):  # Segment based on edges
                self.segment_based_on_edges()
            elif key == ord('s'):  # Save the mask and exit
                # Find contours of the mask
                contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Get bounding box for the largest contour
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    # y_start = max(0, y - 10)
                    # y_end = min(self.image_copy.shape[0], y + h + 10)
                    # x_start = max(0, x - 10)
                    # x_end = min(self.image_copy.shape[1], x + w + 10)

                    y_start = y
                    y_end = y + h
                    x_start = x
                    x_end = x + w

                    cropped_object = self.image_copy[y_start:y_end, x_start:x_end]
                    cropped_mask = self.mask[y_start:y_end, x_start:x_end]
                    # Extract only the region under the mask
                    # Save the result
                    object_cut_path = self.mask_path.replace("mask", "object_cut").replace(".png", ".jpg")
                    cv2.imwrite(object_cut_path, cropped_object)
                    print(f"Cropped object saved at: {object_cut_path}")

                    cv2.imwrite(self.mask_path, cropped_mask)
                    print(f"Mask saved at: {self.mask_path}")
                else:
                    print("No valid contours found for cropping.")
                break


            elif key == ord('z'):  # Undo last action
                self.undo()
            elif key == ord('y'):  # Redo last undone action
                self.redo()
            elif key == ord('q'):  # Quit without saving
                print("Exited without saving.")
                break

        cv2.destroyAllWindows()
        return self.mask

if __name__ == '__main__':

    i = 2
    image_path = f"./input_data/{i}object.jpg"  # Replace with your image path
    mask_path = f"./input_data/{i}mask.png"

    painter = MaskPainter(image_path, mask_path)
    painter.create_mask()
