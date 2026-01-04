import os
import json
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from model import MultiTaskModel

app = Flask(__name__)
CORS(app)

# Configuration
ARTIFACT_DIR = "saved_models"
CKPT_PATH = os.path.join(ARTIFACT_DIR, "multitask_cnn_lstm.pth")
VOCAB_PATH = os.path.join(ARTIFACT_DIR, "vocab.json")
ACTIONS_PATH = os.path.join(ARTIFACT_DIR, "action_labels.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Artifacts
print("Loading artifacts...")
if not os.path.exists(CKPT_PATH):
    print(f"Error: Checkpoint not found at {CKPT_PATH}")

try:
    ckpt = torch.load(CKPT_PATH, map_location=device)
    conf = ckpt["config"]
    
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vdata = json.load(f)
        
    with open(ACTIONS_PATH, "r", encoding="utf-8") as f:
        action_classes = json.load(f)
        
    pad_id = int(conf["pad_id"])
    sos_id = int(conf["sos_id"])
    eos_id = int(conf["eos_id"])
    itos = vdata["itos"]
    
    # Initialize Model
    model = MultiTaskModel(
        num_actions=len(action_classes),
        vocab_size=int(conf["vocab_size"]),
        embed_dim=int(conf["embed_dim"]),
        hidden_dim=int(conf["hidden_dim"]),
        num_layers=int(conf["num_lstm_layers"]),
        pad_id=pad_id,
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    itos = []
    sos_id = 0
    eos_id = 0

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_caption(model, img_t, itos, sos_id, eos_id, max_len=20):
    """Generate caption using greedy decoding."""
    with torch.no_grad():
        # Get image features
        img_feat = model.cnn(img_t)
        
        # Initialize LSTM hidden states from image features
        h0 = torch.tanh(model.caption_head.to_h0(img_feat))
        c0 = torch.tanh(model.caption_head.to_c0(img_feat))
        h = h0.unsqueeze(0).repeat(model.caption_head.num_layers, 1, 1)
        c = c0.unsqueeze(0).repeat(model.caption_head.num_layers, 1, 1)
        
        # Start with SOS token
        cur = torch.tensor([[sos_id]], device=img_t.device)
        out_tokens = []
        
        # Generate tokens one by one
        for _ in range(max_len):
            emb = model.caption_head.embed(cur)
            y, (h, c) = model.caption_head.lstm(emb, (h, c))
            logits = model.caption_head.out(y[:, -1, :])
            nxt = torch.argmax(logits, dim=-1).item()
            
            # Stop if EOS token
            if nxt == eos_id:
                break
            
            out_tokens.append(nxt)
            cur = torch.tensor([[nxt]], device=img_t.device)
        
        # Convert token IDs to words
        words = [itos[i] if i < len(itos) else "<UNK>" for i in out_tokens]
        return " ".join(words)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    try:
        img = Image.open(file).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Action Prediction
            action_logits, _ = model(img_t)
            probs = F.softmax(action_logits, dim=-1).squeeze(0).cpu()
            
            # Get top 5 predictions
            topk = torch.topk(probs, k=min(5, len(action_classes)))
            
            actions = []
            for p, i in zip(topk.values.tolist(), topk.indices.tolist()):
                actions.append({
                    'label': action_classes[i],
                    'score': p
                })
            
            # Caption Generation
            caption = generate_caption(model, img_t, itos, sos_id, eos_id)
        
        return jsonify({
            'actions': actions,
            'caption': caption
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
