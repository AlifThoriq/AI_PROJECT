import tensorflow as tf
import os

# Matikan log sampah TensorFlow biar output bersih
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"\n{'='*40}")
print(f"Versi TensorFlow: {tf.__version__}")
print(f"{'='*40}")

# Cek GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("\n✅ SUKSES! GPU Terdeteksi:")
    for i, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get('device_name', 'Unknown GPU')
        print(f"   [{i}] {name}")
    print("\n🚀 Siap training pakai GTX 1050!")
else:
    print("\n❌ YAH... Masih CPU yang kebaca.")
    print("Coba cek lagi langkah copy-paste cuDNN/Zlib atau Path Environment.")
print(f"{'='*40}\n")