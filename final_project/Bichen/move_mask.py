import cv2
import numpy as np

def move_mask(object_image_path, mask_path, background_image_path, output_image_path, output_mask_path):
    object_image = cv2.imread(object_image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    background_image = cv2.imread(background_image_path)

    if object_image is None or mask is None or background_image is None:
        raise ValueError("Could not load one or more input images.")

    h_bg, w_bg = background_image.shape[:2]
    h_mask, w_mask = mask.shape[:2]
    h_obj, w_obj = object_image.shape[:2]

    overlay = background_image.copy()
    alpha = 0.5  # Transparency for mask visualization

    offset_x, offset_y = 0, 0  # Initial offsets
    start_x, start_y = 0, 0
    dragging = False

    def draw(event, x, y, flags, param):
        nonlocal offset_x, offset_y, start_x, start_y, dragging
        if event == cv2.EVENT_LBUTTONDOWN:  # Start dragging
            start_x, start_y = x, y
            dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and dragging:  # Dragging mask
            offset_x += x - start_x
            offset_y += y - start_y
            start_x, start_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:  # Stop dragging
            dragging = False

    cv2.namedWindow("Move Mask")
    cv2.setMouseCallback("Move Mask", draw)

    while True:
        moved_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)
        # Calculate the mask's position
        x_start = max(0, offset_x)
        y_start = max(0, offset_y)
        x_end = min(w_bg, x_start + w_mask)
        y_end = min(h_bg, y_start + h_mask)

        mask_x_start = max(0, -offset_x)
        mask_y_start = max(0, -offset_y)
        mask_x_end = mask_x_start + (x_end - x_start)
        mask_y_end = mask_y_start + (y_end - y_start)

        # Calculate the overlapping region dimensions
        overlap_width = min(x_end - x_start, mask_x_end - mask_x_start)
        overlap_height = min(y_end - y_start, mask_y_end - mask_y_start)

        # Ensure the slices match the overlapping region
        moved_mask[y_start:y_start + overlap_height, x_start:x_start + overlap_width] = \
            mask[mask_y_start:mask_y_start + overlap_height, mask_x_start:mask_x_start + overlap_width]

        # Overlay the mask onto the background image
        color_mask = np.zeros_like(background_image)
        color_mask[y_start:y_start + overlap_height, x_start:x_start + overlap_width] = (0, 255, 0)  # Green mask
        blended = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)

        cv2.imshow("Move Mask", blended)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save and exit
            # Resize mask and object to match the background dimensions
            resized_mask = cv2.resize(moved_mask, (w_bg, h_bg), interpolation=cv2.INTER_NEAREST)
            resized_object = cv2.resize(object_image, (w_bg, h_bg), interpolation=cv2.INTER_LINEAR)

            # Extract the object part under the mask
            final_object = np.zeros_like(background_image)
            final_object[resized_mask > 0] = resized_object[resized_mask > 0]
            cv2.imwrite(output_mask_path, resized_mask)
            cv2.imwrite(output_image_path, final_object)

            print(f"Saved mask to {output_mask_path} and object to {output_image_path}.")
            break
        elif key == ord('q'):  # Quit without saving
            print("Exited without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    i = 1
    object_image_path = f"./input_data/object{i}.jpg"
    mask_path = f"./input_data/mask{i}.png"
    background_image_path = f"./input_data/background{i}.jpg"
    output_image_path = f"./input_data/object_move{i}.png"
    output_mask_path = f"./input_data/mask_move{i}.png"

    move_mask(object_image_path, mask_path, background_image_path, output_image_path, output_mask_path)
