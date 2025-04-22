import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim


# Ensures UTF-8 encoding for stdout to handle special characters that charmap can't process
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Detects device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Design for the Vanilla Character-level RNN
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size

        # Weight matrices
        self.U = nn.Parameter(torch.randn(hidden_size, input_size, device=device) * 0.01)
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size, device=device) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_size, 1, device=device))
        self.V = nn.Parameter(torch.randn(output_size, hidden_size, device=device) * 0.01)
        self.c = nn.Parameter(torch.zeros(output_size, 1, device=device))

    def forward(self, x, h_prev):
        h = torch.tanh(self.U @ x + self.W @ h_prev + self.b)
        logits = self.V @ h + self.c

        # We feed in the logits instead of using softmax since cross-entropy loss will apply softmax internally
        return logits, h

# Function to generate a character one-hot encoded vector
def one_hot_encode(index, vocab_size=256):
    vec = torch.zeros(vocab_size, 1, device=device)
    vec[index] = 1.0
    return vec

# Model training function
def train(model, data, epochs=20, seq_length=25, lr=0.001, clip_grad=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    print("Starting Training:")

    for epoch in range(1, epochs + 1):
        h_prev = torch.zeros(model.hidden_size, 1, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(data) - seq_length, seq_length):
            inputs = data[i : i + seq_length]
            targets = data[i + 1 : i + seq_length + 1]
            optimizer.zero_grad()
            h = h_prev.detach()
            loss = 0.0

            # Forward pass
            for t in range(seq_length):
                x = one_hot_encode(inputs[t])
                logits, h = model(x, h)
                loss += loss_fn(
                    logits.squeeze().unsqueeze(0),
                    torch.tensor([targets[t]], dtype=torch.long, device=device),
                )

            # Backward pass
            loss.backward()

            # Gradient clipping and parameter update
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            h_prev = h.detach()

        # Average loss for the epoch
        avg_loss = epoch_loss / n_batches if n_batches else float('nan')
        losses.append(avg_loss)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

        # Saves the model every 2 epochs
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    # Plot training loss vs epochs
    plt.figure()
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.show()

    return losses

# Function to generate text using the trained model and a seed
def generate_text(model, seed_char, length=200, temperature=1.0):
    model.to(device)
    model.eval()
    idx = ord(seed_char)
    h = torch.zeros(model.hidden_size, 1, device=device)
    input_vec = one_hot_encode(idx)
    chars = [seed_char]

    with torch.no_grad():
        for _ in range(length):
            logits, h = model(input_vec, h)
            probs = torch.softmax(logits.squeeze() / temperature, dim=0)
            idx = torch.multinomial(probs, num_samples=1).item()
            chars.append(chr(idx))
            input_vec = one_hot_encode(idx)

    return ''.join(chars)

if __name__ == "__main__":
    # Configuration parameters
    DATA_PATH = 'datasets/tmotw.txt'
    MODEL_PATH = None  # Default Model Path so I can easily change it to None or a path
    MODEL_PATH = 'model_epoch_8.pth' # Comment out to train the model
    EPOCHS = 10
    SEQ_LENGTH = 25
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 128
    GENERATE_LENGTH = 200
    SEED_CHAR = 'A'
    TEMPERATURE = 1.0

    # Loads the text data
    with open(DATA_PATH, 'r', encoding='latin-1') as f:
            text = f.read()

    # Converts text to ascii values
    data = [ord(ch) for ch in text]
    vocab_size = 256

    # Initializes the model
    model = CharRNN(vocab_size, HIDDEN_SIZE, vocab_size)

    # Trains or Generates text based on the MODEL_PATH
    if MODEL_PATH is None:
        # Trains the model
        train(model, data, epochs=EPOCHS, seq_length=SEQ_LENGTH, lr=LEARNING_RATE)
    else:
        # Loads the model
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        # Generates sample text
        sample = generate_text(model, seed_char=SEED_CHAR, length=GENERATE_LENGTH, temperature=TEMPERATURE)
        print("\n--- Generated Text Sample ---")
        print(sample)
