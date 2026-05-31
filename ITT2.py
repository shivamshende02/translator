import cv2
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from spelling import convert_text_to_corrected



processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten",use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def crop_and_recognize(image_path, results):
    
 
    image = cv2.imread(image_path)
    final_text = "" 

    

    for idx, (bbox, score) in reversed(list(enumerate(zip(results[0], results[1])))):       
       

        x_coords = [point[0] for point in bbox]  
        y_coords = [point[1] for point in bbox]  

        
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)

       
        cropped_img = image[y1:y2, x1:x2]

        
        if cropped_img is None or cropped_img.size == 0:
            x1-= 10
            y1-= 10
            x2 += 10
            y2 += 10
            cropped_img = image[y1:y2, x1:x2]

        
        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        
        pixel_values = processor(pil_img, return_tensors="pt").pixel_values

        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        
        final_text += text + " "

        
        #cv2.imwrite(f"cropped_{idx}.jpg", cropped_img)
        convert_text_to_corrected(final_text)

    return final_text.strip()




