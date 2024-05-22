from TTS.api import TTS
import torch
from pydub import AudioSegment
import os

def audio_dataset_from_cloned_voices(text_dataset):
    # generated with coquiTTS
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List available 🐸TTS models
    print(TTS().list_models())

    # Init TTS
    yourtts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
    voiceClone = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    #tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False).to(device)
    jennytts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=False).to(device)

    for sid, sample in text_dataset.iterrows():
      if sid < 48:
          continue
      print(f"Generating mp3 for sample {sid}")    
      print(f"Text: {sample['text']}")
      text = sample['text']

      # RP3012_2.wav is the good speaker wav to use; use jenny and your_tts
      #jennytts.tts_to_file(text, file_path=f"../data/audio/sample_jenny_{sid}.wav")
      #yourtts.tts_to_file(text, speaker_wav="../data/audio_samples_for_voice_generation/RP3012_2.wav", language="en", file_path=f"../data/audio/sample_yourTTS_{sid}.wav")
      voiceClone.tts_to_file(
        text,
        speaker_wav=[
            "/Users/tijuana/development/deeplearning/DeepLearning_Project/data/audio_samples_for_voice_generation/RP3012.wav",
            "/Users/tijuana/development/deeplearning/DeepLearning_Project/data/audio_samples_for_voice_generation/RP3012_3.wav",
            "/Users/tijuana/development/deeplearning/DeepLearning_Project/data/audio_samples_for_voice_generation/RP3012_4.wav",
            "/Users/tijuana/development/deeplearning/DeepLearning_Project/data/audio_samples_for_voice_generation/RP3012_5.wav",
            "/Users/tijuana/development/deeplearning/DeepLearning_Project/data/audio_samples_for_voice_generation/RP3012_6.wav",
            "/Users/tijuana/development/deeplearning/DeepLearning_Project/data/audio_samples_for_voice_generation/RP3012_7.wav",
            "/Users/tijuana/development/deeplearning/DeepLearning_Project/data/audio_samples_for_voice_generation/RP3012_8.wav",
                     ],
        file_path=f"../data/audio/voice_clone_nino{sid}.wav",
        language="en",
        split_sentences=True
        )


    return None

def pad_audiofile_to_length(audio, length):
    # Calculate the amount of silence needed
    silence_length = length - len(audio)

    # Create a silence audio segment
    silence = AudioSegment.silent(duration=silence_length)

    # Add the silence to the end of the audio file
    padded_audio = audio + silence

    return padded_audio

def pad_dataset(path_to_audiofiles):
    # Get a list of all .wav files in the directory
    audio_files = [f for f in sorted(os.listdir(path_to_audiofiles), key=lambda x: int(x.split('_')[2].split('.')[0]) if '_' in x and len(x.split('_')) > 2 and x.split('_')[2].split('.')[0].isdigit() else 0) if f.endswith('.wav')]

    # Rest of your code...
    for f in audio_files:
        print(f)
    # Find the length of the longest audio file
    max_length = 0
    for filename in audio_files:
        audio = AudioSegment.from_wav(os.path.join(path_to_audiofiles, filename))
        if len(audio) > max_length:
            max_length = len(audio)

    # Pad each audio file to the length of the longest one and save it
    for i, filename in enumerate(audio_files):

        audio = AudioSegment.from_wav(os.path.join(path_to_audiofiles, filename))
        padded_audio = pad_audiofile_to_length(audio, max_length)
        padded_audio.export(os.path.join(path_to_audiofiles, f"padded/padded_{i}.mp3"), format="mp3")

    return None
    
    