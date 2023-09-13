import torch
import torch.nn as nn
import torch.optim as optim

# Sample neural network architecture
class DQNSnake(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNSnake, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def get_action(state, model, device="cpu"):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state)
    action = torch.argmax(q_values, axis=1).item()
    return action
