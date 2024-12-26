import argparse
import numpy as np
import cv2
from os import path


class MaskMover():
    def __init__(self, image_path, mask_path, object_path):
        self.image = cv2.imread(image_path)
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.object = cv2.imread(object_path)
        self.image_h, self.image_w = self.image.shape[:2]
        self.mask_h, self.mask_w = self.mask.shape[:2]
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.window_name = "Move Mask (Left mouse to move, 's' to save, 'q' to quit)"
        self.target_mask = np.zeros_like(self.image, dtype=np.uint8)
        self.target_object = np.zeros_like(self.image, dtype=np.uint8)

    def move_mask(self):
        def draw_overlay():
            overlay = self.image.copy()
            mask_y1 = max(0, self.offset_y)
            mask_y2 = min(self.image_h, self.offset_y + self.mask_h)
            mask_x1 = max(0, self.offset_x)
            mask_x2 = min(self.image_w, self.offset_x + self.mask_w)
            mask_crop_y1 = max(0, -self.offset_y)
            mask_crop_y2 = mask_crop_y1 + (mask_y2 - mask_y1)
            mask_crop_x1 = max(0, -self.offset_x)
            mask_crop_x2 = mask_crop_x1 + (mask_x2 - mask_x1)
            if mask_y1 < mask_y2 and mask_x1 < mask_x2:
                overlay[mask_y1:mask_y2, mask_x1:mask_x2] = cv2.addWeighted(
                    overlay[mask_y1:mask_y2, mask_x1:mask_x2],
                    0.5,
                    cv2.merge([self.mask, self.mask, self.mask])[mask_crop_y1:mask_crop_y2, mask_crop_x1:mask_crop_x2],
                    0.5,
                    0
                )
            return overlay

        def save_results():
            mask_y1 = max(0, self.offset_y)
            mask_y2 = min(self.image_h, self.offset_y + self.mask_h)
            mask_x1 = max(0, self.offset_x)
            mask_x2 = min(self.image_w, self.offset_x + self.mask_w)
            mask_crop_y1 = max(0, -self.offset_y)
            mask_crop_y2 = mask_crop_y1 + (mask_y2 - mask_y1)
            mask_crop_x1 = max(0, -self.offset_x)
            mask_crop_x2 = mask_crop_x1 + (mask_x2 - mask_x1)
            if mask_y1 < mask_y2 and mask_x1 < mask_x2:
                self.target_mask[mask_y1:mask_y2, mask_x1:mask_x2] = cv2.merge([self.mask, self.mask, self.mask])[
                    mask_crop_y1:mask_crop_y2, mask_crop_x1:mask_crop_x2
                ]
                self.target_object[mask_y1:mask_y2, mask_x1:mask_x2] = self.object[
                    mask_crop_y1:mask_crop_y2, mask_crop_x1:mask_crop_x2
                ]

        def on_mouse(event, x, y, flags, param):
            nonlocal last_mouse_pos
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                last_mouse_pos = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                dx, dy = x - last_mouse_pos[0], y - last_mouse_pos[1]
                self.offset_x += dx
                self.offset_y += dy
                last_mouse_pos = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, on_mouse)
        last_mouse_pos = (0, 0)

        while True:
            overlay = draw_overlay()
            cv2.imshow(self.window_name, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Quit
                break
            elif key == ord("s"):  # Save results
                save_results()
                target_mask_path = path.join(path.dirname(image_path), f"{i}target_mask.png")
                target_object_path = path.join(path.dirname(image_path), f"{i}target_object.png")
                cv2.imwrite(target_mask_path, self.target_mask)
                cv2.imwrite(target_object_path, self.target_object)
                print(f"Saved target mask to: {target_mask_path}")
                print(f"Saved target object to: {target_object_path}")
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    i = 3
    image_path = f"./input_data/{i}background.jpg"  # Replace with your image path
    mask_path = f"./input_data/{i}mask.png"
    object_path = f"./input_data/{i}object_cut.jpg"


    mm = MaskMover(image_path, mask_path, object_path)
    mm.move_mask()
