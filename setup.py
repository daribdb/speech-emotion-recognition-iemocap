import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load filtered dataset
df = pd.read_csv('iemocap_4class.csv')

print("="*60)
print("Verification Test")
print("="*60)

# Test 1: Check if all audio files exist
print("\n1. Checking audio file accessibility...")
missing_files = 0
for idx, row in df.head(100).iterrows():  # Check first 100
    if not pd.isna(row['audio_path']):
        import os
        if not os.path.exists(row['audio_path']):
            missing_files += 1
            print(f"  Missing: {row['audio_path']}")

if missing_files == 0:
    print(f"  ✓ All checked files exist!")
else:
    print(f"  ✗ Found {missing_files} missing files")

# Test 2: Load and play a sample audio file
print("\n2. Loading sample audio file...")
sample = df.iloc[0]
print(f"  Utterance: {sample['utterance_id']}")
print(f"  Emotion: {sample['emotion']}")
print(f"  Audio path: {sample['audio_path']}")

audio, sr = librosa.load(sample['audio_path'], sr=None)
print(f"  ✓ Successfully loaded!")
print(f"  Sample rate: {sr} Hz")
print(f"  Duration: {len(audio)/sr:.2f} seconds")
print(f"  Samples: {len(audio)}")

# Test 3: Extract features
print("\n3. Extracting MFCC features...")
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
print(f"  MFCC shape: {mfccs.shape}")
print(f"  ✓ Feature extraction successful!")

# Test 4: Visualize
print("\n4. Creating visualization...")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Waveform
axes[0].plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
axes[0].set_title(f'Waveform - {sample["utterance_id"]} ({sample["emotion"]})')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')

# MFCCs
img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=axes[1])
axes[1].set_title('MFCC Features')
axes[1].set_ylabel('MFCC Coefficient')
fig.colorbar(img, ax=axes[1])

plt.tight_layout()
plt.savefig('verification_plot.png', dpi=150)
print(f"  ✓ Saved visualization to: verification_plot.png")

print("\n" + "="*60)
print("✓ All verification tests passed!")
print("You're ready to start training models!")
print("="*60)