from flask import Flask, jsonify, request
from model import DQNSnake, get_action
import torch

app = Flask(__name__)

# Load your trained model here
model_path = "path_to_your_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQNSnake(input_dim=400, output_dim=4).to(device)  # Example dimensions, adjust as per your model's architecture
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@app.route('/get_action', methods=['POST'])
def predict_action():
    data = request.get_json()
    state = data.get("state", [])
    action = get_action(state, model, device)
    return jsonify({"action": action})

if __name__ == '__main__':
    app.run(debug=True)
