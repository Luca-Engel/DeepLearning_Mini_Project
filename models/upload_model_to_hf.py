from datasets import Dataset, Audio
import pandas as pd
import os
from tqdm import tqdm


def upload_model_to_huggingface(model):
    """
    Uploads the model to Hugging Face Models Hub

    :param model: the model to save on huggingface hub
    """

    model.push_to_hub(
        repo_id="DL-Project/DL_Audio_Hatespeech_wav2vec",
        private=True,
    )


######################################################################
######## log in with huggingface-cli before running this file ########
######################################################################
if __name__ == "__main__":
    upload_model_to_huggingface()

