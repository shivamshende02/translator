from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import subprocess
import os
from marathi_t import translate_to_marathi

# Define paths
script_path = r"D:\Translate\Translator\detection.py"
model_path = r"D:\Translate\Translator\Models\text_detection_en_ppocrv3_2023may.onnx"
backend_target = "0"

# Create an upload folder if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']

    # Save the image to the upload folder
    file_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(file_path)

    # Run the script with subprocess
    try:
        result = subprocess.run(
            ["python", script_path, "--input", file_path, "--model", model_path, "-bt", backend_target],
            check=True,
            capture_output=True,  
            text=True  
        )
        return jsonify({'result': result.stdout.strip()}) 
    except subprocess.CalledProcessError as e:
        print("Error in subprocess execution:", e.stderr)  
        return jsonify({'error': e.stderr.strip(), 'debug': str(e)}), 500  

@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid input"}), 400
        
        user_text = data["text"]
        response_text = translate_to_marathi(user_text)
        
        return jsonify({"response": response_text})
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": "Server error", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)
