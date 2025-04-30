from flask import Flask, request, jsonify
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Load your model (assuming you have a .pth model file)
model = torch.load('D:\Code research\AI Model\Lingomaster_Model\model')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Get the files from the request
        recording = request.files['recording']  # The recording file
        word = request.form['word']  # The selected word

        # Save the recording to a temporary file
        recording_path = os.path.join('temp', recording.filename)
        recording.save(recording_path)

        # Process the recording and evaluate it using your model
        # (You would need to add your model evaluation logic here)
        # Example: result = model.evaluate(recording_path, word)
        result = "Your pronunciation is good!"  # Placeholder result

        return jsonify({"result": result, "word": word, "file": recording_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
