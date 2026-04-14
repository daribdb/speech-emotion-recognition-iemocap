import os
import glob
import pandas as pd
import re

IEMOCAP_PATH = '/Users/darigaborasheva/Desktop/IEMOCAP_full_release'

def create_full_dataset_index():
    """
    Create a complete index of all utterances with their labels
    """
    all_data = []
    
    for session in range(1, 6):
        print(f"Processing Session {session}...")
        
        session_path = os.path.join(IEMOCAP_PATH, f'Session{session}')
        eval_path = os.path.join(session_path, 'dialog', 'EmoEvaluation')
        
        # Process each evaluation file
        for eval_file in sorted(glob.glob(os.path.join(eval_path, '*.txt'))):
            dialog_name = os.path.basename(eval_file).replace('.txt', '')
            
            with open(eval_file, 'r') as f:
                for line in f:
                    if line.startswith('['):
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            # Parse line
                            time_range = parts[0].strip('[]').split(' - ')
                            start_time = float(time_range[0])
                            end_time = float(time_range[1])
                            
                            utterance_id = parts[1].strip()
                            emotion = parts[2].strip()
                            
                            # Parse VAD
                            vad_str = parts[3].strip('[]').split(', ')
                            valence = float(vad_str[0])
                            activation = float(vad_str[1])
                            dominance = float(vad_str[2])
                            
                            # Determine audio path
                            audio_path = find_audio_file(utterance_id, session)
                            
                            # Extract speaker info
                            speaker_id = utterance_id.split('_')[0]  # e.g., 'Ses01F'
                            speaker_gender = speaker_id[-1]  # 'F' or 'M'
                            turn_speaker = utterance_id[-4]  # 'F' or 'M'
                            
                            # Determine if improvised or scripted
                            interaction_type = 'improvised' if 'impro' in utterance_id else 'scripted'
                            
                            all_data.append({
                                'utterance_id': utterance_id,
                                'session': session,
                                'dialog': dialog_name,
                                'emotion': emotion,
                                'valence': valence,
                                'activation': activation,
                                'dominance': dominance,
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': end_time - start_time,
                                'speaker_id': speaker_id,
                                'speaker_gender': speaker_gender,
                                'turn_speaker': turn_speaker,
                                'interaction_type': interaction_type,
                                'audio_path': audio_path,
                                'audio_exists': audio_path is not None
                            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\nDataset index created!")
    print(f"Total utterances: {len(df)}")
    print(f"Utterances with audio: {df['audio_exists'].sum()}")
    
    return df

def find_audio_file(utterance_id, session):
    """
    Find the audio file path for a given utterance ID
    """
    # Extract dialog info from utterance_id
    # Example: Ses01F_impro01_F000
    match = re.search(r'Ses\d{2}[FM]_(impro\d+|script\d+_\d+)', utterance_id)
    if not match:
        return None
    
    dialog_name = match.group(0)  # e.g., 'Ses01F_impro01'
    
    # Construct audio path
    audio_filename = f'{utterance_id}.wav'
    audio_path = os.path.join(
        IEMOCAP_PATH,
        f'Session{session}',
        'sentences',
        'wav',
        dialog_name,
        audio_filename
    )
    
    if os.path.exists(audio_path):
        return audio_path
    else:
        return None

# Create the index
df = create_full_dataset_index()

# Save to CSV
output_file = 'iemocap_full_index.csv'
df.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")

# Show statistics
print("\n" + "="*60)
print("Dataset Statistics")
print("="*60)
print(f"\nTotal utterances: {len(df)}")
print(f"\nEmotion distribution:")
print(df['emotion'].value_counts())
print(f"\nInteraction type:")
print(df['interaction_type'].value_counts())
print(f"\nGender distribution:")
print(df['speaker_gender'].value_counts())
print(f"\nUtterances per session:")
print(df['session'].value_counts().sort_index())

# Show first few rows
print("\n" + "="*60)
print("Sample Data")
print("="*60)
print(df.head(10))