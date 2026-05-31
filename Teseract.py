import cv2
import pytesseract
from PIL import Image
import numpy as np
from spelling import convert_text_to_corrected

def crop_and_recognize(image_path, results):
    # Read the main image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return ""

    h, w, _ = image.shape
    final_text = ""  

    for idx, (bbox, score) in reversed(list(enumerate(zip(results[0], results[1])))):
        x_coords = [point[0] for point in bbox]  # Extract x values
        y_coords = [point[1] for point in bbox]  # Extract y values

        # Get bounding rectangle (clamped within image bounds)
        x1, y1 = max(0, min(x_coords) - 1), max(0, min(y_coords) - 1)
        x2, y2 = min(w, max(x_coords) + 1), min(h, max(y_coords) + 1)

        # Crop the image based on bounding rectangle
        cropped_img = image[y1:y2, x1:x2]

        # Ensure cropped_img is valid
        if cropped_img is None or cropped_img.size == 0:
            x1-= 10
            y1-= 10
            x2 += 10
            y2 += 10
            cropped_img = image[y1:y2, x1:x2]

          
        
        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        # OCR processing
        text = pytesseract.image_to_string(pil_img, lang='eng')
        final_text += text + " "

        # Save cropped image for debugging
        #cv2.imwrite(f"cropped_{idx}.jpg", cropped_img)
    
    
    convert_text_to_corrected(final_text)
    return final_text.strip()
