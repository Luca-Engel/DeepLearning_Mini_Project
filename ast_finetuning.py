import torch
from sklearn.model_selection import KFold

from data.fetch_dataset_from_hf import fetch_dataset_from_huggingface
from huggingface_hub import login
from representation_learner import create_label_id_dicts, preprocess_function, compute_metrics
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from models.upload_model_to_hf import upload_model_to_huggingface
import os

os.environ["HF_TOKEN"] = "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"
login("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")

print("fetching dataset")
dataset = fetch_dataset_from_huggingface()
print(dataset)

dataset = dataset.remove_columns(["hate_speech_score", "text"])  # removes unused columns
label2id, id2label = create_label_id_dicts(dataset)
print(label2id)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593")  # MIT/ast-finetuned-audioset-10-10-0.4593
print("feature_extractor downloaded...")
encoded_dataset = preprocess_function(dataset, feature_extractor)

print(encoded_dataset)
train_dataset = encoded_dataset["train"].train_test_split(test_size=0.2)
print(train_dataset)

num_labels = len(label2id)
model = AutoModelForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=num_labels, label2id=label2id, id2label=id2label,
    ignore_mismatched_sizes=True
)


training_args = TrainingArguments(
    # use_cpu=True,
    do_train=True,
    output_dir="models/finetuned_ast",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_only_model=True,  # only saves model to checkpoints
    learning_rate=3e-5,  # 3e-5
    weight_decay=0.005,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_dir="tensorboard",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=True,
    seed=42,
    fp16=True,
    
    hub_private_repo=True,
    hub_model_id="DL-Project/DL_Audio_Hatespeech_ast_trainer_push",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=train_dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

print("starting training...")
trainer.train()


print("eval scores:\n", trainer.evaluate())

print("test set scores:\n", trainer.evaluate(encoded_dataset["test"]))

upload_model_to_huggingface(model, repo_id="DL-Project/DL_Audio_Hatespeech_ast", commit_message="full dataset, lr: 3e-5, batch_size: 16, epochs: 2")

trainer.push_to_hub(
    commit_message="ast trainer push best model after 5 epochs fp16"
)

