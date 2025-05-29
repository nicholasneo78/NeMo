import torch
import soundfile as sf
from nemo.collections.asr.models import EncDecSpeakerLabelModel

# 1. Load model
model_path = "/models/nemo_test/TitaNet-Finetune/2025-04-30_06-19-01-ReduceLROnPlateau-with-ES/checkpoints/TitaNet-Finetune.nemo"
model = EncDecSpeakerLabelModel.restore_from(restore_path=model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
model.eval()

# 2. Load and prepare audio
audio_file = "/datasets/mms_lid/test_split/mms/test/CHDIR_495_2022-06-29_08-00046697-00053645-00.wav"
signal, sr = sf.read(audio_file)

if sr != 16000:
    raise ValueError(f"Sample rate must be 16 kHz, got {sr}")

# Convert to tensor, batch it
signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
length_tensor = torch.tensor([signal_tensor.shape[1]], dtype=torch.int64).to(device)

# 3. Forward pass
with torch.no_grad():
    logits, _ = model(input_signal=signal_tensor, input_signal_length=length_tensor)
    logits = logits[0]  # Remove batch dimension if needed
    probs = torch.softmax(logits, dim=-1)

print(model.cfg)

# Get label list from config
label_list = model.cfg.train_ds.labels

# Get predicted label index and label name
print(probs)
predicted_index = torch.argmax(probs, dim=-1).item()
predicted_label = label_list[predicted_index]

print(f"Predicted index: {predicted_index}")
print(f"Confidence: {probs[predicted_index].item()}")
print(f"Predicted language: {predicted_label}")

