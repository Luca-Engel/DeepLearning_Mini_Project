import torch
from data.fetch_dataset_from_hf import fetch_dataset_from_huggingface
from huggingface_hub import login
from representation_learner import create_label_id_dicts, preprocess_function, compute_metrics
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from models.upload_model_to_hf import upload_model_to_huggingface
import os

seed = 42
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)

login("replace with read token")
dataset = fetch_dataset_from_huggingface()

dataset = dataset.remove_columns(["hate_speech_score", "text"])  # removes unused columns
label2id, id2label = create_label_id_dicts(dataset)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base")  # "facebook/wav2vec2-base"

encoded_dataset = preprocess_function(dataset, feature_extractor)

num_labels = len(label2id)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)

# model.train()

training_args = TrainingArguments(
    # use_cpu=True,
    do_train=True,
    output_dir="models/finetuned_wav2vec",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,  # 3e-5
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=25,
    warmup_ratio=0.1,
    logging_dir="tensorboard",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    seed=seed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

print(trainer.evaluate())

login("replace with write token")
upload_model_to_huggingface(model)


