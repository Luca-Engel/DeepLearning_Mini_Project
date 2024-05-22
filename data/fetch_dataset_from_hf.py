from datasets import load_dataset
from huggingface_hub import login
import soundfile as sf


def _remove_invalid_samples(dataset):
    valid_indices_train = []
    invalid_indices_train = []
    max_length = 0
    for i, sample in enumerate(dataset):
        try:
            # Attempt to read the audio data
            waveform_sample = sample["audio_waveform"]["array"]
            valid_indices_train.append(i)
            max_length = max(max_length, len(waveform_sample))
        except sf.LibsndfileError:
            invalid_indices_train.append(i)

    dataset = dataset.select(valid_indices_train)

    return dataset


def fetch_dataset_from_huggingface():
    """
    Fetches the audio dataset from Hugging Face Datasets Hub
    :return: dataset object
    """

    # If the dataset is gated/private, make sure you have run huggingface-cli login
    dataset = load_dataset("DL-Project/DL_Audio_Hatespeech_Dataset") #, streaming=True)

    #dataset["train"] = _remove_invalid_samples(dataset["train"])
    #dataset["test"] = _remove_invalid_samples(dataset["test"])

    return dataset




if __name__ == "__main__":
    dataset = fetch_dataset_from_huggingface()

    print(dataset)
    print(dataset["train"][0])
    print(dataset["train"][0]["audio_waveform"])

    print(dataset["test"][0])
    print(dataset["test"][0]["audio_waveform"])

