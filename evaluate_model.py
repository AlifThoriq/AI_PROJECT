import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import librosa
import cv2
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === PATH KONFIGURASI ===
DATASET_ROOT = r"C:\Users\Darkt\Downloads\dataset_local"  # folder dataset
CKPT_PATH = r"E:\AI_Project\outputs\checkpoints\best_model.h5"  # model terbaik
IMG_W, IMG_H = 64, 64
N_MELS = 64
SR = 22050
MAX_AUDIO_SEC = 2.0
BATCH_SIZE = 32

# === UTILITAS ===
def retrive_data_recursive(directory):
    json_files = []
    audio_files = []
    for dirpath, _, filenames in os.walk(directory):
        for fn in filenames:
            lf = fn.lower()
            full = os.path.join(dirpath, fn)
            if lf.endswith(".json"):
                json_files.append(full)
            elif lf.endswith((".webm", ".ogg", ".wav", ".mp3")):
                audio_files.append(full)
    return json_files, audio_files

def find_audio_for_base(base, search_root=DATASET_ROOT):
    for ext in (".webm", ".ogg", ".wav", ".mp3"):
        p = Path(search_root) / (base + ext)
        if p.exists():
            return str(p)
    matches = list(Path(search_root).rglob(base + ".*"))
    for m in matches:
        if m.suffix.lower() in (".webm", ".ogg", ".wav", ".mp3"):
            return str(m)
    return None

def load_audio(path, sr=SR, max_sec=MAX_AUDIO_SEC):
    y, _ = librosa.load(path, sr=sr, mono=True)
    max_len = int(sr * max_sec)
    if len(y) > max_len:
        start = np.random.randint(0, len(y) - max_len + 1)
        y = y[start:start + max_len]
    else:
        y = np.pad(y, (0, max(0, max_len - len(y))), mode='constant')
    return y

def make_mel(y, sr=SR, n_mels=N_MELS, img_w=IMG_W, img_h=IMG_H):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    S_img = (S_norm * 255).astype(np.uint8)
    resized = cv2.resize(S_img, (img_w, img_h))
    return resized[..., np.newaxis].astype(np.float32) / 255.0

# === DATAFRAME METADATA ===
json_paths = [str(p) for p in Path(DATASET_ROOT).rglob("*.json")]
rows = []
for jp in json_paths:
    try:
        j = json.load(open(jp, 'r', encoding='utf-8'))
        base = Path(jp).stem
        rows.append({
            "filename": base,
            "json_path": jp,
            "status": j.get("status"),
        })
    except Exception:
        continue
df_all = pd.DataFrame(rows)

if 'audio_path' not in df_all.columns:
    df_all['audio_path'] = df_all['filename'].apply(lambda b: find_audio_for_base(b, DATASET_ROOT))

df_meta = df_all[(df_all['audio_path'].notnull()) & (df_all['status'].notnull())].copy().reset_index(drop=True)

# === FILTER LABEL ===
wanted = ['healthy', 'symptomatic', 'COVID-19']
df_meta = df_meta[df_meta['status'].isin(wanted)].reset_index(drop=True)
label_map = {lab: i for i, lab in enumerate(sorted(df_meta['status'].unique()))}
inv_label_map = {v: k for k, v in label_map.items()}
df_meta['label'] = df_meta['status'].map(label_map)

# === SPLIT DATA ===
from sklearn.model_selection import train_test_split
_, test_df = train_test_split(df_meta, test_size=0.3, stratify=df_meta['label'], random_state=42)

# === SEQUENCE GENERATOR ===
from tensorflow.keras.utils import Sequence

class AudioSequence(Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, shuffle=False, augment=False):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = self.df.iloc[batch_idx]
        X = np.zeros((len(batch), IMG_H, IMG_W, 1), dtype=np.float32)
        y = np.zeros((len(batch),), dtype=np.int32)
        for i, (_, row) in enumerate(batch.iterrows()):
            path = row['audio_path']
            audio = load_audio(path)
            spec = make_mel(audio)
            X[i] = spec
            y[i] = int(row['label'])
        return X, keras.utils.to_categorical(y, num_classes=len(label_map))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# === LOAD MODEL ===
model = keras.models.load_model(CKPT_PATH)

# === GENERATOR TEST ===
test_gen = AudioSequence(test_df, batch_size=BATCH_SIZE, shuffle=False, augment=False)

# === PREDIKSI ===
y_true = []
y_pred = []

for X_batch, y_batch in test_gen:
    preds = model.predict(X_batch)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# === EVALUASI ===
accuracy = np.mean(y_true == y_pred)
print("Test Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=inv_label_map.values(), yticklabels=inv_label_map.values(), cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, target_names=inv_label_map.values())
print(report)
