import pandas as pd

# Load the full index
df = pd.read_csv('iemocap_full_index.csv')

print("Original dataset:")
print(f"Total utterances: {len(df)}")
print(f"\nEmotion distribution:")
print(df['emotion'].value_counts())

# Step 1: Keep only the target emotions
target_emotions = ['neu', 'hap', 'exc', 'sad', 'ang']
df_filtered = df[df['emotion'].isin(target_emotions)].copy()

print(f"\n\nAfter filtering to {target_emotions}:")
print(f"Total utterances: {len(df_filtered)}")
print(f"\nEmotion distribution:")
print(df_filtered['emotion'].value_counts())

# Step 2: Merge 'hap' and 'exc'
df_filtered['emotion'] = df_filtered['emotion'].replace('exc', 'hap')

print(f"\n\nAfter merging 'exc' into 'hap':")
print(f"Total utterances: {len(df_filtered)}")
print(f"\nEmotion distribution:")
print(df_filtered['emotion'].value_counts())

# Step 3: Create numeric labels
emotion_to_label = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
label_to_emotion = {v: k for k, v in emotion_to_label.items()}

df_filtered['label'] = df_filtered['emotion'].map(emotion_to_label)

# Step 4: Keep only rows with valid audio files
df_filtered = df_filtered[df_filtered['audio_exists'] == True].copy()

print(f"\n\nAfter removing utterances without audio:")
print(f"Total utterances: {len(df_filtered)}")

# Step 5: Extract simplified speaker ID for cross-validation
# From 'Ses01F' extract '01F' (session + gender)
df_filtered['speaker'] = df_filtered['speaker_id'].str[-3:]

print(f"\n\nSpeakers in dataset:")
print(sorted(df_filtered['speaker'].unique()))
print(f"Total speakers: {df_filtered['speaker'].nunique()}")

# Save filtered dataset
output_file = 'iemocap_4class.csv'
df_filtered.to_csv(output_file, index=False)
print(f"\n\nFiltered dataset saved to: {output_file}")

# Final statistics
print("\n" + "="*60)
print("Final Dataset Statistics (Following Paper Guidelines)")
print("="*60)
print(f"\nTotal samples: {len(df_filtered)}")
print(f"\nClass distribution:")
for label in sorted(df_filtered['label'].unique()):
    emotion = label_to_emotion[label]
    count = (df_filtered['label'] == label).sum()
    percentage = (count / len(df_filtered)) * 100
    print(f"  {emotion} (label {label}): {count:4d} ({percentage:5.2f}%)")

print(f"\nInteraction type:")
print(df_filtered['interaction_type'].value_counts())

print(f"\nAverage duration: {df_filtered['duration'].mean():.2f} seconds")
print(f"Total audio duration: {df_filtered['duration'].sum() / 3600:.2f} hours")

print(f"\nSpeakers: {sorted(df_filtered['speaker'].unique())}")