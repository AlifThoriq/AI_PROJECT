import os
import json
import numpy as np
import pandas as pd
import librosa
import cv2
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

# === 🔴 SETTING KHUSUS GPU GTX 1050 (WAJIB ADA) ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Memory Growth Enabled: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ GPU tidak terdeteksi, berjalan di CPU (Lambat).")

# === KONFIGURASI PATH ===
DATASET_ROOT = r"C:\Users\Darkt\Downloads\dataset_local"
OUTPUT_DIR = r"E:\AI_Project\outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "history"), exist_ok=True)

STATE_PATH = os.path.join(OUTPUT_DIR, "history", "training_state.json")

print("TensorFlow version:", tf.__version__)

# === 🛠️ CUSTOM METRIC: MACRO F1-SCORE (FIXED) ===
class MacroF1Score(keras.metrics.Metric):
    def __init__(self, name='f1_macro', **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        # REVISI: Tambahkan shape=(3,) karena kita punya 3 kelas
        self.tp = self.add_weight(name='tp', shape=(3,), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(3,), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(3,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.argmax(y_true, axis=-1)
        
        y_true = K.one_hot(y_true, 3) 
        y_pred = K.one_hot(y_pred, 3)

        self.tp.assign_add(K.sum(y_true * y_pred, axis=0))
        self.fp.assign_add(K.sum((1 - y_true) * y_pred, axis=0))
        self.fn.assign_add(K.sum(y_true * (1 - y_pred), axis=0))

    def result(self):
        p = self.tp / (self.tp + self.fp + K.epsilon())
        r = self.tp / (self.tp + self.fn + K.epsilon())
        f1 = 2 * p * r / (p + r + K.epsilon())
        return K.mean(f1)

    def reset_state(self):
        self.tp.assign(K.zeros(3))
        self.fp.assign(K.zeros(3))
        self.fn.assign(K.zeros(3))

# === FUNGSI UTILITAS ===
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

json_files, audio_files = retrive_data_recursive(DATASET_ROOT)

# === MEMBANGUN DATAFRAME METADATA ===
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

if 'audio_path' not in df_all.columns:
    df_all['audio_path'] = df_all['filename'].apply(lambda b: find_audio_for_base(b, DATASET_ROOT))

df_meta = df_all[(df_all['audio_path'].notnull()) & (df_all['status'].notnull())].copy().reset_index(drop=True)

# === FILTER LABEL ===
wanted = ['healthy', 'symptomatic', 'COVID-19']
df_meta = df_meta[df_meta['status'].isin(wanted)].reset_index(drop=True)
label_map = {lab: i for i, lab in enumerate(sorted(df_meta['status'].unique()))}
inv_label_map = {v: k for k, v in label_map.items()}
df_meta['label'] = df_meta['status'].map(label_map)
print("Setelah filter label:", df_meta['status'].value_counts())

# === SPLIT TRAIN/VAL/TEST ===
train_df, temp_df = train_test_split(df_meta, test_size=0.30, stratify=df_meta['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
print(f"Data Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# === OVERSAMPLING MINORITAS ===
def oversample_minority(df):
    df_healthy = df[df['status'] == 'healthy']
    df_symp = df[df['status'] == 'symptomatic']
    df_covid = df[df['status'] == 'COVID-19']

    target = 5000 
    
    df_symp_resampled = df_symp.sample(target, replace=True, random_state=42)
    df_covid_resampled = df_covid.sample(target, replace=True, random_state=42)
    df_healthy_resampled = df_healthy.sample(target, replace=False, random_state=42) 

    df_balanced = pd.concat([df_healthy_resampled, df_symp_resampled, df_covid_resampled])
    return df_balanced.sample(frac=1, random_state=42)

train_df_balanced = oversample_minority(train_df)
print("Distribusi Training (Balanced):")
print(train_df_balanced['status'].value_counts())

# === AUDIO CONFIG ===
IMG_W = 128
IMG_H = 128
N_MELS = 128
SR = 22050
MAX_AUDIO_SEC = 4.0

# === SPEC-AUGMENT ===
def spec_augment(spec, num_freq_masks=2, num_time_masks=2):
    spec = spec.copy()
    freq_mask_size = IMG_H // 8
    time_mask_size = IMG_W // 5
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_size)
        f0 = np.random.randint(0, spec.shape[0] - f)
        spec[f0:f0+f, :] = 0
    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_size)
        t0 = np.random.randint(0, spec.shape[1] - t)
        spec[:, t0:t0+t] = 0
    return spec

# === FUNGSI AUDIO ===
def load_audio(path, sr=SR, max_sec=MAX_AUDIO_SEC):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading audio {path}: {e}")
        y = np.zeros(int(sr * max_sec))
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

# === AUGMENTASI ===
def augment_audio_strong(y, sr=SR):
    if np.random.rand() < 0.5:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-3, 3))
    if np.random.rand() < 0.5:
        y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))
    if np.random.rand() < 0.5:
        noise_amp = 0.01 * np.random.uniform()
        y = y + noise_amp * np.random.randn(len(y))
    y = np.clip(y, -1.0, 1.0)
    return y

def augment_audio(y, sr=SR):
    if np.random.rand() < 0.5:
        shift = int(sr * np.random.uniform(-0.1, 0.1))
        y = np.roll(y, shift)
    if np.random.rand() < 0.3:
        noise_amp = 0.005 * np.random.uniform()
        y = y + noise_amp * np.random.randn(len(y))
    y = np.clip(y, -1.0, 1.0)
    return y

# === GENERATOR ===
class AudioSequence(Sequence):
    def __init__(self, df, batch_size=32, shuffle=True, augment=False):
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
            status = row['status']
            audio = load_audio(path)
            if self.augment:
                if status != 'healthy':
                    audio = augment_audio_strong(audio)
                else:
                    audio = augment_audio(audio)
            spec = make_mel(audio)
            if self.augment:
                spec = spec_augment(spec.squeeze()).reshape(IMG_H, IMG_W, 1)
            X[i] = spec
            y[i] = int(row['label'])
        return X, keras.utils.to_categorical(y, num_classes=len(label_map))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# === ARSITEKTUR MODEL ===
def build_model(input_shape=(IMG_H, IMG_W, 1), n_classes=len(label_map)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inp, out)

# === 🔧 CUSTOM CALLBACK: SIMPAN STATUS EPOCH ===
class EpochSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        with open(STATE_PATH, 'w') as f:
            json.dump({"last_epoch": current_epoch}, f)

# === 🚀 LOGIKA RESUME TRAINING ===

ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.h5")

f1_macro = MacroF1Score() 
initial_epoch = 0

if os.path.exists(ckpt_path):
    print(f"\n🔄 Checkpoint ditemukan di: {ckpt_path}")
    
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, 'r') as f:
                state = json.load(f)
                initial_epoch = state.get("last_epoch", 0)
            print(f"📂 Melanjutkan dari EPOCH KE-{initial_epoch + 1}...")
        except:
            print("⚠️ Gagal membaca history state.")
    
    print("⏳ Memuat model...")
    model = keras.models.load_model(ckpt_path, custom_objects={'MacroF1Score': MacroF1Score}, compile=False)
    print("✅ Model dimuat!")

else:
    print(f"\n🆕 Memulai training dari AWAL.")
    model = build_model()

print("⚙️ Mengompilasi model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', f1_macro]
)

model.summary()


# === PERSIAPAN DATA GENERATOR (BATCH SIZE 16) ===
batch_size = 16 

train_gen = AudioSequence(train_df_balanced, batch_size=batch_size, shuffle=True, augment=True)
val_gen   = AudioSequence(val_df, batch_size=batch_size, shuffle=False, augment=False)
test_gen  = AudioSequence(test_df, batch_size=batch_size, shuffle=False, augment=False)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor='val_f1_macro', mode='max',
        save_best_only=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, 
        restore_best_weights=True, verbose=1
    ),
    EpochSaver()
]

print(f"\n🚀 Training GPU Dimulai! (Batch Size: {batch_size})")
epochs = 40 
history = model.fit(
    train_gen,
    validation_data=val_gen,
    initial_epoch=initial_epoch,
    epochs=epochs,
    callbacks=callbacks
)

print("\nTraining selesai.")
print("Model terbaik disimpan di:", ckpt_path)


# === EVALUASI MODEL ===
print("\n=== Mengevaluasi Model dengan Test Set ===")

model.load_weights(ckpt_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_macro])

test_metrics = model.evaluate(test_gen, verbose=1)
print(f"Test Loss: {test_metrics[0]:.4f}")
print(f"Test Accuracy: {test_metrics[1]:.4f}")
print(f"Test F1-Macro: {test_metrics[2]:.4f}")

print("\nMembuat prediksi untuk Confusion Matrix...")
y_true = test_df['label'].values
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = [inv_label_map[i] for i in range(len(label_map))]
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

cm_fig_path = os.path.join(OUTPUT_DIR, "final_confusion_matrix.png")
plt.savefig(cm_fig_path)
print(f"Confusion Matrix disimpan di: {cm_fig_path}")