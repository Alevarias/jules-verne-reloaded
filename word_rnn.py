import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Ensures UTF-8 encoding for stdout to handle special characters that charmap can't process
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Detect device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Design for the Word-level RNN
class WordRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WordRNN, self).__init__()
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

# Function to generate a 5000 long one-hot encoded vector
def one_hot_encode(index, vocab_size=5000):
    vec = torch.zeros(vocab_size, 1, device=device)
    vec[index] = 1.0
    return vec

# Model training function
def train(model, data, epochs=20, seq_length=25, lr=0.001, clip_grad=5):
    model.train()
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
                x = one_hot_encode(inputs[t], vocab_size)
                logits, h = model(x, h)
                loss += loss_fn(logits.squeeze().unsqueeze(0),
                                torch.tensor([targets[t]], dtype=torch.long, device=device))
            
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

        # Save the model every epoch
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"wrnn_model_epoch_{epoch}.pth")

    # Plot training loss vs epochs
    plt.figure()
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.show()

    return losses

# Generation function (word-level)
def generate_text(model, seed_word, length=50, temperature=1.0):
    model.eval()
    h = torch.zeros(model.hidden_size, 1, device=device)
    # get seed index, default to unknown if missing
    seed_idx = vocab.get(seed_word, vocab.get("[UNK]", 0))
    input_vec = one_hot_encode(seed_idx, vocab_size)
    words = [seed_word]

    with torch.no_grad():
        for _ in range(length):
            logits, h = model(input_vec, h)
            probs = torch.softmax(logits.squeeze() / temperature, dim=0)
            idx = torch.multinomial(probs, num_samples=1).item()
            word = idx2token.get(idx, "[UNK]")
            words.append(word)
            input_vec = one_hot_encode(idx, vocab_size)

    return ' '.join(words)

if __name__ == "__main__":
    # Config
    DATA_PATH = 'datasets/tmotw.txt'
    MODEL_PATH = None  # or 'model_epoch_8.pth'\
    MODEL_PATH = 'model_epoch_8.pth' # Comment out to train the model
    EPOCHS = 5
    SEQ_LENGTH = 25
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 128
    GENERATE_LENGTH = 50
    SEED_WORD = 'The'
    TEMPERATURE = 1.0

    # 1. Read raw text
    with open(DATA_PATH, 'r', encoding='latin-1') as f:
        raw_text = f.read().lower()

    # 2. Train a WordLevel tokenizer
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(vocab_size=5000, special_tokens=["[UNK]"])
    tokenizer.train([DATA_PATH], trainer)

    # 3. Extract vocab and mappings
    vocab = tokenizer.get_vocab()  # token -> index
    vocab_size = len(vocab)
    idx2token = {i: t for t, i in vocab.items()}

    # 4. Encode entire text as indices
    encoded = tokenizer.encode(raw_text)
    data = encoded.ids

    # 5. Init model
    model = WordRNN(vocab_size, HIDDEN_SIZE, vocab_size)

    # 6. Train or load+generate
    if MODEL_PATH is None:
        train(model, data, epochs=EPOCHS,
              seq_length=SEQ_LENGTH, lr=LEARNING_RATE)
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        sample = generate_text(model, SEED_WORD, length=GENERATE_LENGTH, temperature=TEMPERATURE)
        print("\n--- Generated Text Sample ---")
        print(sample)
