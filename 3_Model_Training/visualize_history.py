# 3_Model_Training/visualize_history.py
import os
import json
import matplotlib.pyplot as plt

MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "6_Models_and_Outputs")
HISTORY_JSON = os.path.join(MODEL_OUTPUT_DIR, "history.json")

if not os.path.exists(HISTORY_JSON):
    raise FileNotFoundError(f"History file not found: {HISTORY_JSON}. Run train_model.py first.")

with open(HISTORY_JSON, "r") as f:
    hist = json.load(f)

# Plot accuracy
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(hist.get("accuracy", []), label="train_accuracy")
plt.plot(hist.get("val_accuracy", []), label="val_accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot loss
plt.subplot(1,2,2)
plt.plot(hist.get("loss", []), label="train_loss")
plt.plot(hist.get("val_loss", []), label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()