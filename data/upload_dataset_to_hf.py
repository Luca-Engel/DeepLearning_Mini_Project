from datasets import Dataset, Audio
import pandas as pd
import os
from tqdm import tqdm


def upload_dataset_to_huggingface(
        audio_dir="data/audio/jenny/padded",
        annotations_file="data/text/preprocessed.csv",
        test_size_fraction=0.2
):
    """
    Creates and uploads the dataset to Hugging Face Datasets Hub

    :param audio_dir: directory containing the audio files
    :param annotations_file: file containing the annotations
    :param test_size_fraction: train-test split fraction (size of the test set)
    :return: dataset object
    """
    annotations_df = pd.read_csv(annotations_file)
    dataset_dict = {"audio_waveform": [], "hate_speech_score": [], "label": [], "text": []}

    largest_audio_id = 0
    for filename in os.listdir(audio_dir):
        if filename.startswith("padded_") and filename.endswith(".mp3"):
            audio_id = int(filename.split("_")[1].split(".")[0])
            largest_audio_id = max(largest_audio_id, audio_id)

    print(f"Largest audio ID: {largest_audio_id}")

    for audio_id, row in annotations_df.iterrows():
        if audio_id > largest_audio_id:
            break

        audio_path = os.path.join(audio_dir, f"padded_{audio_id}.mp3")
        # waveform, sr = librosa.load(audio_path, sr=None)  # Load audio file as waveform
        hate_speech_score = row["hate_speech_score"]
        label = annotations_df.loc[audio_id, "hate_speech_label"]
        text = annotations_df.loc[audio_id, "text"]

        dataset_dict["audio_waveform"].append(audio_path)
        dataset_dict["hate_speech_score"].append(hate_speech_score)
        dataset_dict["label"].append(label)
        dataset_dict["text"].append(text)

    # Hugging Face Dataset object will contain dict as waveform ("array" will have the entries) because of cast_column
    dataset = Dataset.from_dict(dataset_dict).cast_column("audio_waveform", Audio(sampling_rate=16000))
    dataset = dataset.class_encode_column("label") # 0 = hatespeech, 1 = non-hatespeech"
    dataset = dataset.train_test_split(test_size=test_size_fraction)#, shuffle=True, seed=0)
    print(dataset["train"][0]["audio_waveform"])
    print(dataset["train"][0])
    print()

    print(dataset["test"][0]["audio_waveform"])
    print(dataset["test"][0])
    print()
    print(dataset)


    dataset.push_to_hub(
        repo_id="DL-Project/DL_Audio_Hatespeech_Dataset",
        private=True,
    )


######################################################################
######## log in with huggingface-cli before running this file ########
######################################################################
if __name__ == "__main__":
    upload_dataset_to_huggingface()

