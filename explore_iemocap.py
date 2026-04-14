import os
import glob
from collections import Counter

# Set your IEMOCAP path
IEMOCAP_PATH = '/Users/darigaborasheva/Desktop/IEMOCAP_full_release'

def explore_sessions():
    """Print overview of all sessions"""
    print("="*60)
    print("IEMOCAP Dataset Overview")
    print("="*60)
    
    for session in range(1, 6):
        session_path = os.path.join(IEMOCAP_PATH, f'Session{session}')
        
        # Count audio files
        wav_path = os.path.join(session_path, 'sentences', 'wav')
        wav_files = glob.glob(os.path.join(wav_path, '*', '*.wav'))
        
        # Count annotation files
        eval_path = os.path.join(session_path, 'dialog', 'EmoEvaluation')
        eval_files = glob.glob(os.path.join(eval_path, '*.txt'))
        
        print(f"\nSession {session}:")
        print(f"  Audio files: {len(wav_files)}")
        print(f"  Annotation files: {len(eval_files)}")

def explore_emotions():
    """Count all emotion labels in the dataset"""
    print("\n" + "="*60)
    print("Emotion Distribution Across All Sessions")
    print("="*60)
    
    all_emotions = []
    
    for session in range(1, 6):
        eval_path = os.path.join(IEMOCAP_PATH, f'Session{session}', 'dialog', 'EmoEvaluation')
        
        for eval_file in glob.glob(os.path.join(eval_path, '*.txt')):
            with open(eval_file, 'r') as f:
                for line in f:
                    if line.startswith('['):
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            emotion = parts[2]
                            all_emotions.append(emotion)
    
    # Count emotions
    emotion_counts = Counter(all_emotions)
    
    print(f"\nTotal utterances: {len(all_emotions)}")
    print("\nEmotion breakdown:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_emotions)) * 100
        print(f"  {emotion}: {count:4d} ({percentage:5.2f}%)")
    
    return emotion_counts

def show_sample_utterances(session=1, dialog='impro01', num_samples=5):
    """Show sample utterances from a specific dialog"""
    print("\n" + "="*60)
    print(f"Sample Utterances from Session{session} - {dialog}")
    print("="*60)
    
    # Read annotation file
    eval_file = os.path.join(
        IEMOCAP_PATH, 
        f'Session{session}', 
        'dialog', 
        'EmoEvaluation',
        f'Ses0{session}F_{dialog}.txt'
    )
    
    print(f"\nReading: {eval_file}\n")
    
    count = 0
    with open(eval_file, 'r') as f:
        for line in f:
            if line.startswith('[') and count < num_samples:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    time_range = parts[0]
                    utterance_id = parts[1]
                    emotion = parts[2]
                    vad = parts[3]
                    
                    # Check if audio file exists
                    audio_path = os.path.join(
                        IEMOCAP_PATH,
                        f'Session{session}',
                        'sentences',
                        'wav',
                        f'Ses0{session}F_{dialog}',
                        f'{utterance_id}.wav'
                    )
                    exists = "✓" if os.path.exists(audio_path) else "✗"
                    
                    print(f"{count+1}. {utterance_id}")
                    print(f"   Time: {time_range}")
                    print(f"   Emotion: {emotion}")
                    print(f"   VAD: {vad}")
                    print(f"   Audio exists: {exists}")
                    print(f"   Path: {audio_path}")
                    print()
                    
                    count += 1

def check_audio_file(utterance_id, session):
    """Check if a specific audio file exists and get its info"""
    import librosa
    
    # Determine dialog type and name
    if 'impro' in utterance_id:
        dialog_type = 'impro'
    else:
        dialog_type = 'script'
    
    # Extract dialog number (e.g., 'impro01' or 'script01_1')
    import re
    match = re.search(r'(impro\d+|script\d+_\d+)', utterance_id)
    if match:
        dialog_name = match.group(1)
    else:
        print("Could not parse dialog name")
        return
    
    # Construct path
    audio_path = os.path.join(
        IEMOCAP_PATH,
        f'Session{session}',
        'sentences',
        'wav',
        f'Ses0{session}F_{dialog_name}',  # Assuming female, adjust if needed
        f'{utterance_id}.wav'
    )
    
    if not os.path.exists(audio_path):
        # Try male
        audio_path = os.path.join(
            IEMOCAP_PATH,
            f'Session{session}',
            'sentences',
            'wav',
            f'Ses0{session}M_{dialog_name}',
            f'{utterance_id}.wav'
        )
    
    if os.path.exists(audio_path):
        print(f"✓ Found audio file:")
        print(f"  Path: {audio_path}")
        
        # Load and analyze
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Samples: {len(audio)}")
    else:
        print(f"✗ Audio file not found")
        print(f"  Searched: {audio_path}")

# Run exploration
if __name__ == "__main__":
    # Explore all sessions
    explore_sessions()
    
    # Count emotions
    emotion_counts = explore_emotions()
    
    # Show samples
    show_sample_utterances(session=1, dialog='impro01', num_samples=5)
    
    # Check a specific file
    print("\n" + "="*60)
    print("Checking Specific Audio File")
    print("="*60)
    check_audio_file('Ses01F_impro01_F000', session=1)