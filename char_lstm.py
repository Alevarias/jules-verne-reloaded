import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Ensures UTF-8 encoding for stdout to handle special characters that charmap can't process
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Detect device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        # Single-layer LSTMCell
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        # Fully-connected layer to map hidden state to logits over vocabulary
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, states):
        # x: (vocab_size, 1) -> reshape to (1, vocab_size)
        x = x.view(1, -1)
        h_prev, c_prev = states
        # LSTMCell expects input (batch=1, input_size) and (h_prev, c_prev)
        h_next, c_next = self.lstm_cell(x, (h_prev, c_prev))
        # Produce logits: (1, output_size)
        logits = self.fc(h_next)
        # Return in shape (output_size, 1) and next states
        return logits.t(), (h_next, c_next)


def one_hot_encode(index, vocab_size=256):
    vec = torch.zeros(vocab_size, 1, device=device)
    vec[index] = 1.0
    return vec


def train(model, data,
          epochs=20, seq_length=25,
          lr=1e-3, clip_grad=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    print("Starting LSTM Training:")
    for epoch in range(1, epochs + 1):
        # Initialize hidden and cell states to zeros: (batch=1, hidden_size)
        h_prev = torch.zeros(1, model.hidden_size, device=device)
        c_prev = torch.zeros(1, model.hidden_size, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(data) - seq_length, seq_length):
            inputs = data[i:i + seq_length]
            targets = data[i + 1:i + seq_length + 1]
            optimizer.zero_grad()
            h, c = h_prev.detach(), c_prev.detach()
            loss = 0.0

            # Truncated BPTT
            for t in range(seq_length):
                x = one_hot_encode(inputs[t])
                logits, (h, c) = model(x, (h, c))
                # logits: (vocab_size, 1) -> squeeze & unsqueeze for CE
                loss += loss_fn(
                    logits.squeeze().unsqueeze(0),
                    torch.tensor([targets[t]], dtype=torch.long, device=device)
                )

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            h_prev, c_prev = h.detach(), c.detach()

        avg_loss = epoch_loss / n_batches if n_batches else float('nan')
        losses.append(avg_loss)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"char_lstm_epoch_{epoch}.pth")

    # Plot
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Char-LSTM Training Loss')
    plt.show()
    return losses


def generate_text(model, seed_char, length=200, temperature=1.0):
    model.to(device)
    model.eval()
    # Initialize states
    h = torch.zeros(1, model.hidden_size, device=device)
    c = torch.zeros(1, model.hidden_size, device=device)
    # First input
    idx = ord(seed_char)
    input_vec = one_hot_encode(idx)
    chars = [seed_char]

    with torch.no_grad():
        for _ in range(length):
            logits, (h, c) = model(input_vec, (h, c))
            probs = torch.softmax(logits.squeeze() / temperature, dim=0)
            idx = torch.multinomial(probs, num_samples=1).item()
            chars.append(chr(idx))
            input_vec = one_hot_encode(idx)
    return ''.join(chars)


if __name__ == "__main__":
    # Configuration parameters
    DATA_PATH = 'datasets/tmotw.txt'
    MODEL_PATH = None
    MODEL_PATH = 'char_lstm_epoch_10.pth' # Comment out to train the model
    EPOCHS = 10
    SEQ_LENGTH = 25
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 128
    GENERATE_LENGTH = 200
    SEED_CHAR = 'A'
    TEMPERATURE = 1.0

    # Load data
    with open(DATA_PATH, 'r', encoding='latin-1') as f:
        text = f.read()
    data = [ord(ch) for ch in text]

    # Initialize and train/load model
    vocab_size = 256
    model = CharLSTM(vocab_size, HIDDEN_SIZE, vocab_size)
    if MODEL_PATH:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Loaded CharLSTM from {MODEL_PATH}")

        # Generate sample text
        sample = generate_text(model, seed_char=SEED_CHAR,
                            length=GENERATE_LENGTH,
                            temperature=TEMPERATURE)
        print("\n--- Generated Text Sample ---")
        print(sample)



    else:
        train(model, data, epochs=EPOCHS,
              seq_length=SEQ_LENGTH,
              lr=LEARNING_RATE)

    
