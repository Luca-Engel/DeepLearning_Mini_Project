import datasets
import copy
from gtts import gTTS, lang
from io import BytesIO
import data.fetch_dataset_from_hf
from pydub import AudioSegment
import os
import sys
#import all the modules that we will need to use
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import TTS

def load_dataset():
    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', split='train[:10000]') # TODO: adapt that number depending on what amount of the dataset we want
    #dataset = fetch_dataset_from_huggingface()
    return dataset

def split_dataset(dataset):
    # hate_dataset = dataset.train_test_split(test_size=0.2, seed=1)
    # train_dataset, test_dataset = hate_dataset['train'], hate_dataset['test']
    # return train_dataset, test_dataset
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    return train_dataset, test_dataset

def mp3_dataset_from_text(text_dataset):
    # TODO: automize mp3 generation, and add columns for audio infos
    #assert False, f"{lang.tts_langs()}"
    dataset = copy.deepcopy(text_dataset)
    
    accents = {
        "com.au": "Australia",
        "co.uk": "United Kingdom",
        "us": "United States",
        "ca": "Canada",
        "co.in": "India",
        "ie": "Ireland",
        "co.za": "South Africa"
    }
    
    for sid, sample in dataset.iterrows():
        if sid < 0:
            continue
        print(f"Generating mp3 for sample {sid}")    
        print(f"Text: {sample['text']}")
        text = sample['text']
        accent = list(accents.keys())[sid % len(accents)]
        tts = gTTS(text=text, lang='en', tld=accent)
        
        filename = f'sample_{sid}'
        with open(f'data/audio/{filename}.mp3', 'wb') as f:
            tts.write_to_fp(f)
            
        sample['mp3_file'] = filename
        sample['accent'] = accent

    # TODO: maybe add code to save the dataset to a file
    
    return dataset


def mp3_dataset_from_text_coquiTTS(text_dataset):
    dataset = copy.deepcopy(text_dataset)

    # Initialize the ModelManager
    model_manager = ModelManager()

    # Download the desired TTS model
    model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/glow-tts")
#ek1/tacotron2 better

    # Download the default vocoder for the TTS model
    voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])

    # Process the first text with TTS
    syn = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        vocoder_checkpoint=voc_path,
        vocoder_config=voc_config_path
    )

    for sid, sample in dataset.iterrows():
      if sid > 5:
          continue
      print(f"Generating mp3 for sample {sid}")    
      print(f"Text: {sample['text']}")
      text = sample['text']
      audio = syn.tts(text)

      syn.save_wav(audio, f'data/audio/audio_{sid}.wav')

    return None

def audio_dataset_from_cloned_voices(text_dataset):
    # generated with coquiTTS
    dataset = copy.deepcopy(text_dataset)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

    tts.tts_to_file(text="I hate these fucking faggots",
                file_path="output.wav",
                speaker_wav=["deeplearning/DeepLearning_Project/data/audio_samples_for_voice_generation/RP3012.wav"],
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
    # Load all audio files and find the length of the longest one
    audio_files = [AudioSegment.from_mp3(os.path.join(path_to_audiofiles, f)) for f in sorted(os.listdir(path_to_audiofiles), key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x and x.split('_')[1].split('.')[0].isdigit() else 0) if f.endswith('.mp3')]
    max_length = max(len(audio) for audio in audio_files)

    # Pad all audio files to the length of the longest one
    padded_audio_files = [pad_audiofile_to_length(audio, max_length) for audio in audio_files]

    # Save the files back to the directory
    for i, audio in enumerate(padded_audio_files):
        audio.export(os.path.join(path_to_audiofiles, f"padded_{i}.mp3"), format="mp3")
   
    return None
    

if __name__ == '__main__':
    dataset = load_dataset()
    train_dataset, test_dataset = split_dataset(dataset)
    print(dataset)
    print(train_dataset[0])
    print(train_dataset[0]["audio_waveform"]["array"])