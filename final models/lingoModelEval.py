from flask import Flask, request, jsonify
import torch
import torchaudio
import torchaudio.transforms as T
from final_models.lingo_model2 import SpeechModel
import os
import torch.nn.functional as F
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Initialize the MFCC transform and VAD (Voice Activity Detection)
transform = T.MFCC(sample_rate=16000, n_mfcc=13)
vad = T.Vad(sample_rate=16000)

def preprocess_waveform(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    waveform = vad(waveform)  # Apply VAD if needed

    return waveform

def extract_mfcc(audio_path):
    waveform = preprocess_waveform(audio_path)
    mfcc = transform(waveform).squeeze(0)
    mfcc = mfcc[:, :300] if mfcc.size(1) > 300 else torch.nn.functional.pad(mfcc, (0, 300 - mfcc.size(1)))
    return mfcc.unsqueeze(0)

def predict(audio_input, model):
    waveform = preprocess_waveform(audio_input)
    mfcc = transform(waveform)
    mfcc = mfcc.squeeze(0).unsqueeze(0)

    if mfcc.abs().sum() == 0:
        print("⚠️ Empty or silent input detected. Skipping prediction.")
        return 0.0

    model.eval()
    with torch.no_grad():
        output = model(mfcc)
    return output.item()

def calculate_similarity(predicted_score, target_score=1.0):
    similarity = 100 * (1 - abs(predicted_score - target_score))
    return max(0, min(100, similarity))

def calculate_cosine_similarity(predicted_score, target_score):
    predicted_tensor = torch.tensor([predicted_score])
    target_tensor = torch.tensor([target_score])
    cosine_similarity = F.cosine_similarity(predicted_tensor.unsqueeze(0), target_tensor.unsqueeze(0))
    return (cosine_similarity.item() + 1) * 50

def calculate_scaled_similarity(predicted_score, min_score=0.0, max_score=1.0):
    normalized = (predicted_score - min_score) / (max_score - min_score)
    similarity = max(0, min(100, normalized * 100))
    return similarity

@app.route('/')
def evaluate():
    try:
        # Log the incoming request data
        print(f"Request data: {request.form}")
        print(f"Request files: {request.files}")

        # Check for missing data
        if 'audio' not in request.files:
            print("Missing audio file")
            return "Missing audio file", 400
        if 'selected_word' not in request.form:
            print("Missing selected word")
            return "Missing selected word", 400
        
        audio_file = request.files['audio']
        selected_word = request.form['selected_word']

        # just for debug
        print(f"Received word: {selected_word}")
        print(f"Received file: {audio_file.filename}")
        os.makedirs("temp", exist_ok=True)
        
        filename = secure_filename(audio_file.filename)
        file_path = os.path.join("temp", filename)
        

        # Generate the model path based on the selected word
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # go one level up
        model_path = os.path.join(base_dir, "model", f"{selected_word}_speech_model.pth")
        print(f"Looking for model at: {model_path}")
        if not os.path.exists(model_path):
           print("Model not found.")
           return jsonify({"error": f"Model for '{selected_word}' not found"}), 400
        
        # Load the model dynamically
        model = SpeechModel(input_dim=13)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Save the recording to a temporary file
        filename = secure_filename(audio_file.filename)
        file_path = os.path.join("temp", filename)
        audio_file.save(file_path)

        # Evaluate the recording
        predicted_score = predict(file_path, model)
        similarity = calculate_similarity(predicted_score)

        cosine_sim = calculate_cosine_similarity(predicted_score, target_score=1)
        scale_sim = calculate_scaled_similarity(predicted_score)

        # Calculate blended similarity score
        blended_sim = (cosine_sim + similarity) / 2

        # Return the evaluation results as JSON
        result = {
            "predicted_score": predicted_score,
            "similarity": similarity,
            "cosine_similarity": cosine_sim,
            "scaled_similarity": scale_sim,
            "blended_similarity": blended_sim,
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
