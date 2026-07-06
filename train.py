import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np
import os
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from model import Alice, Bob, Eve

# -----------------------------
# Hyperparameters
# -----------------------------
MSG_LEN = 64
KEY_LEN = 64
BATCH_SIZE = 256      # if GPU supports it
# otherwise keep 256

LR = 8e-4


EPOCHS = 1000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# Models
# -----------------------------

alice = Alice(MSG_LEN, KEY_LEN).to(DEVICE)
bob = Bob(MSG_LEN, KEY_LEN).to(DEVICE)
eve = Eve(MSG_LEN, KEY_LEN).to(DEVICE)

# -----------------------------
# Optimizers
# -----------------------------

opt_ab = optim.Adam(
    list(alice.parameters()) +
    list(bob.parameters()),
    lr=LR
)

opt_eve = optim.Adam(
    eve.parameters(),
    lr=LR
)

# -----------------------------
# Loss Function
# -----------------------------

criterion = nn.L1Loss(reduction="mean")

# -----------------------------
# History Arrays
# -----------------------------

# Accuracy History
bob_history = []
eve_history = []
gap_history = []

# Loss History
bob_loss_history = []
eve_loss_history = []
ab_loss_history = []

# -----------------------------
# Data Generator
# -----------------------------

def generate_batch():

    msg = torch.randint(
        0,
        2,
        (BATCH_SIZE, MSG_LEN),
        device=DEVICE
    ).float()

    key = torch.randint(
        0,
        2,
        (BATCH_SIZE, KEY_LEN),
        device=DEVICE
    ).float()

    msg = msg * 2 - 1
    key = key * 2 - 1

    return msg, key


# -----------------------------
# Accuracy Function
# -----------------------------

def bit_accuracy(pred, target):

    pred_bits = torch.sign(pred)

    return (
        (pred_bits == target)
        .float()
        .mean()
        .item()
    )


# -----------------------------
# Training Loop
# -----------------------------
RESULT_FILE = "training_results.txt"

with open(RESULT_FILE, "w") as f:
    f.write("="*70 + "\n")
    f.write("NEURAL CRYPTOGRAPHY TRAINING RESULTS\n")
    f.write("="*70 + "\n\n")

    f.write("Hyperparameters\n")
    f.write(f"Epochs      : {EPOCHS}\n")
    f.write(f"Batch Size  : {BATCH_SIZE}\n")
    f.write(f"Message Len : {MSG_LEN}\n")
    f.write(f"Key Len     : {KEY_LEN}\n")
    f.write(f"Learning Rate : {LR}\n")
    f.write("\n")

best_gap = -1

bob_acc = 0.0
eve_acc = 0.0
security_gap = 0.0

for epoch in range(EPOCHS):

    start = time.time()

    # -----------------------------
    # Curriculum Adversarial Learning
    # -----------------------------

    # Initial curriculum
    # Curriculum Learning

    if epoch < 250:

       AB_STEPS = 2
       EVE_STEPS = 1

    elif epoch < 500:

       AB_STEPS = 2
       EVE_STEPS = 2

    else:

       AB_STEPS = 3
       EVE_STEPS = 2
# ---------- Feedback Controller ----------

   
    epoch_bob_loss = 0
    epoch_ab_loss = 0
    epoch_eve_loss = 0

    # -------------------------
    # Train Alice + Bob
    # -------------------------

    for p in eve.parameters():
        p.requires_grad = False

    for _ in range(AB_STEPS):

        msg, key = generate_batch()

        cipher = alice(msg, key)

        bob_out = bob(cipher, key)

        eve_out = eve(cipher)

        bob_err = criterion(bob_out,msg)

        eve_err = criterion(eve_out,msg)

        target = 1.0

        loss_ab = bob_err + ((target-eve_err)**2)/(target**2)
        epoch_bob_loss += bob_err.item()
        epoch_ab_loss += loss_ab.item()

        opt_ab.zero_grad()

        loss_ab.backward()

        torch.nn.utils.clip_grad_norm_(

            list(alice.parameters()) +
            list(bob.parameters()),

            max_norm=1.0
        )

        opt_ab.step()

    for p in eve.parameters():
        p.requires_grad = True
    # -------------------------
    # Train Eve
    # -------------------------

    for p in alice.parameters():
        p.requires_grad = False

    for p in bob.parameters():
        p.requires_grad = False

    for _ in range(EVE_STEPS):

        msg, key = generate_batch()

        with torch.no_grad():
            cipher = alice(msg, key)

        eve_out = eve(cipher)

        eve_loss = criterion(eve_out, msg)

        epoch_eve_loss += eve_loss.item()

        opt_eve.zero_grad()

        eve_loss.backward()

        torch.nn.utils.clip_grad_norm_(

            eve.parameters(),

            max_norm=1.0
        )

        opt_eve.step()

    for p in alice.parameters():
        p.requires_grad = True

    for p in bob.parameters():
        p.requires_grad = True
   
    # -------------------------
    # Evaluation
    # -------------------------

    epoch_bob_loss /= AB_STEPS
    epoch_ab_loss /= AB_STEPS
    epoch_eve_loss /= EVE_STEPS

    bob_loss_history.append(epoch_bob_loss)
    ab_loss_history.append(epoch_ab_loss)
    eve_loss_history.append(epoch_eve_loss)

    msg, key = generate_batch()

    alice.eval()
    bob.eval()
    eve.eval()

    with torch.no_grad():

        cipher = alice(msg, key)

        bob_out = bob(cipher, key)

        eve_out = eve(cipher)

    alice.train()
    bob.train()
    eve.train()

    bob_acc = bit_accuracy(bob_out, msg)

    eve_acc = bit_accuracy(eve_out, msg)

    security_gap = bob_acc - eve_acc
    
    # Moving average of last 10 epochs

    if len(gap_history) >= 10:
        avg_gap = np.mean(gap_history[-10:])
    else:
        avg_gap = security_gap
        
     # Adaptive Feedback


    if epoch > 100 and epoch % 20 == 0:

      if avg_gap < 0.10:
        AB_STEPS = min(AB_STEPS + 2, 5)

      elif avg_gap < 0.20:
        AB_STEPS = min(AB_STEPS + 1, 5)

      elif avg_gap > 0.35:
        EVE_STEPS = min(EVE_STEPS + 1, 5)

      if eve_acc > 0.75:
        AB_STEPS = min(AB_STEPS + 1, 5)

      if bob_acc < 0.95:
        AB_STEPS = min(AB_STEPS + 1, 5)

    if (
    bob_acc > 0.95
    and eve_acc < 0.60
    and security_gap > 0.30
):

        best_gap = security_gap

        torch.save(
            {
                "alice": alice.state_dict(),
                "bob": bob.state_dict(),
                "eve": eve.state_dict()
            },
            "best_model.pth"
        )

    bob_history.append(bob_acc)
    eve_history.append(eve_acc)
    gap_history.append(security_gap)

    elapsed = time.time() - start

    with open(RESULT_FILE, "a") as f:
      f.write(
        f"Epoch {epoch+1:04d} | "
        f"Bob={bob_acc:.4f} | "
        f"Eve={eve_acc:.4f} | "
        f"Gap={security_gap:.4f} | "
        f"BobLoss={epoch_bob_loss:.4f} | "
        f"EveLoss={epoch_eve_loss:.4f} | "
        f"ABLoss={epoch_ab_loss:.4f} | "
        f"AB_STEPS={AB_STEPS} | "
        f"EVE_STEPS={EVE_STEPS}\n"
    )

    
# -----------------------------
# Save Final Model
# -----------------------------

torch.save(
    {
        "alice": alice.state_dict(),
        "bob": bob.state_dict(),
        "eve": eve.state_dict()
    },
    "neural_crypto_64bit.pth"
)

print("\nModel saved successfully!")
with open(RESULT_FILE, "a") as f:

    f.write("\n")
    f.write("="*70 + "\n")
    f.write("FINAL RESULTS\n")
    f.write("="*70 + "\n")

    f.write(f"Final Bob Accuracy : {bob_acc:.4f}\n")
    f.write(f"Final Eve Accuracy : {eve_acc:.4f}\n")
    f.write(f"Security Gap       : {security_gap:.4f}\n")
    f.write(f"Best Security Gap  : {best_gap:.4f}\n")

    f.write("\nFiles Generated\n")
    f.write("-----------------\n")
    f.write("best_model.pth\n")
    f.write("neural_crypto_64bit.pth\n")
    f.write("accuracy_curve.png\n")
    f.write("loss_curve.png\n")
    f.write("history.npz\n")

# -----------------------------
# Plot Results
# -----------------------------

window = 15

def smooth(values):
    return np.convolve(
        values,
        np.ones(window) / window,
        mode="same"
    )

plt.figure(figsize=(10, 6))

plt.plot(smooth(bob_history), label="Bob Accuracy")

plt.plot(eve_history, label="Eve Accuracy")
plt.plot(gap_history, label="Security Gap")

plt.xlabel("Epoch")

plt.ylabel("Score")

plt.title("Neural Cryptography Training")

plt.legend()

plt.grid(True)

plt.savefig("accuracy_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(10,6))

plt.plot(range(1, EPOCHS+1), smooth(bob_loss_history), label="Bob Loss")
plt.plot(range(1, EPOCHS+1), smooth(eve_loss_history), label="Eve Loss")
plt.plot(range(1, EPOCHS+1), smooth(ab_loss_history), label="Alice+Bob Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Loss Curves")

plt.legend()

plt.grid(True)
plt.savefig("loss_curve.png", dpi=300)
plt.close()

np.savez(

    "history.npz",

    bob_acc=bob_history,

    eve_acc=eve_history,

    gap=gap_history,

    bob_loss=bob_loss_history,

    eve_loss=eve_loss_history,

    ab_loss=ab_loss_history

)