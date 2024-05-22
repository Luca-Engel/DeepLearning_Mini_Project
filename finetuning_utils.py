import evaluate
import numpy as np
import torch.nn as nn
from datasets import load_dataset, load_metric, Audio
from transformers import AutoFeatureExtractor, TrainingArguments, Trainer
from sklearn.model_selection import KFold


def preprocess_function(dataset, feature_extractor):
    def prepare_dataset(example):
        audio_arrays = [x["array"] for x in example["audio_waveform"]]
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
        )
        return inputs
    
    dataset = dataset.map(prepare_dataset, remove_columns="audio_waveform", batched=True)
    
    return dataset


def create_label_id_dicts(dataset):
    label_names = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()

    # 0 = hatespeech, 1 = non-hatespeech
    for i, label_name in enumerate(label_names):
        # label2id[label_name] = str(1-i)
        # id2label[str(1-i)] = label_name
        label2id[label_name] = str(i)
        id2label[str(i)] = label_name

    assert "non" not in id2label["0"], "Non-hatespeech is not supposed to have label 0"

    return label2id, id2label


def compute_train_metrics(eval_pred):
    accuracy = load_metric("accuracy")
    recall = load_metric("recall")
    precision = load_metric("precision")
    f1 = load_metric("f1")
    
    predictions = np.argmax(eval_pred.predictions, axis=1)
    label_ids = eval_pred.label_ids

    return {
        **accuracy.compute(predictions=predictions, references=label_ids),
        **recall.compute(predictions=predictions, references=label_ids),
        **precision.compute(predictions=predictions, references=label_ids),
        **f1.compute(predictions=predictions, references=label_ids)
    }


def compute_eval_metrics(predictions, labels):
    # Load evaluation metrics
    accuracy = load_metric("accuracy")
    recall = load_metric("recall")
    precision = load_metric("precision")
    f1 = load_metric("f1")

    return {
        **accuracy.compute(predictions=predictions, references=labels),
        **recall.compute(predictions=predictions, references=labels),
        **precision.compute(predictions=predictions, references=labels),
        **f1.compute(predictions=predictions, references=labels)
    }


def finetune_model(model, encoded_dataset, training_args, feature_extractor, total_epochs=20, use_cross_val=True):
    def train_fold(model, training_args, train_dataset, eval_dataset, feature_extractor):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=feature_extractor,
            compute_metrics=compute_train_metrics
        )
        trainer.train()
        metrics = trainer.evaluate()
        return metrics, model

        
    if use_cross_val:
        n_splits = 5  # number of folds
        kf = KFold(n_splits=5, shuffle=True, random_state=training_args.seed)

        training_args_new = TrainingArguments(
            num_train_epochs=total_epochs / n_splits,     # Modify the number of epochs
            do_train=True,
            output_dir=training_args.output_dir,
            overwrite_output_dir=training_args.overwrite_output_dir,
            evaluation_strategy=training_args.evaluation_strategy,
            save_strategy=training_args.save_strategy,
            save_only_model=training_args.save_only_model,
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            warmup_ratio=training_args.warmup_ratio,
            logging_dir=training_args.logging_dir,
            logging_steps=training_args.logging_steps,
            load_best_model_at_end=training_args.load_best_model_at_end,
            metric_for_best_model=training_args.metric_for_best_model,
            push_to_hub=training_args.push_to_hub,
            seed=training_args.seed, 
            #max_steps=training_args.max_steps
            hub_private_repo=training_args.hub_private_repo,
            hub_model_id=training_args.hub_model_id
        )
        
        fold_metrics = []
        for fold, (train_indices, eval_indices) in enumerate(kf.split(encoded_dataset['train'])):
            print(f"Fold: {fold+1}")

            # Select train and validation sets for this fold
            train_dataset = encoded_dataset['train'].select(indices=train_indices)
            eval_dataset = encoded_dataset['train'].select(indices=eval_indices)

            print(f"Training Fold {fold + 1}...")

            metrics, model = train_fold(model, training_args_new, train_dataset, eval_dataset, feature_extractor)
            fold_metrics.append(metrics)

        # Aggregate and print metrics
        avg_metrics = {metric: sum(m[metric] for m in fold_metrics) / n_splits for metric in fold_metrics[0]}
        print("Average metrics:", avg_metrics)
    
    else:
        training_args_new = TrainingArguments(
            num_train_epochs=total_epochs,     # Modify the number of epochs
            do_train=True,
            output_dir=training_args.output_dir,
            overwrite_output_dir=training_args.overwrite_output_dir,
            evaluation_strategy=training_args.evaluation_strategy,
            save_strategy=training_args.save_strategy,
            save_only_model=training_args.save_only_model,
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            warmup_ratio=training_args.warmup_ratio,
            logging_dir=training_args.logging_dir,
            logging_steps=training_args.logging_steps,
            load_best_model_at_end=training_args.load_best_model_at_end,
            metric_for_best_model=training_args.metric_for_best_model,
            push_to_hub=training_args.push_to_hub,
            seed=training_args.seed,
            
            hub_private_repo=training_args.hub_private_repo,
            hub_model_id=training_args.hub_model_id
        )
        
        trainer = Trainer(
            model=model,
            args=training_args_new,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            tokenizer=feature_extractor,
            compute_metrics=compute_train_metrics,
        )
        
        trainer.train()
        trainer.evaluate()
    
    return model, trainer


def get_predictions_labels_and_loss(model, dataset):
    loss_fn = nn.CrossEntropyLoss()

    predictions = []
    labels = []
    losses = []
    for example in dataset:
        input_values = torch.tensor(example['input_values']).unsqueeze(0)  # Add unsqueeze to add batch dimension
        label = example['label']
        with torch.no_grad():
            outputs = model(input_values)  # Assuming the model takes input_values as input
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()  # Consider the last dimension for argmax
        loss = loss_fn(outputs.logits, torch.tensor([label]))  # Compute loss using the specified loss function
        predictions.append(predicted_label)
        labels.append(label)
        losses.append(loss.item())
    return predictions, labels, losses