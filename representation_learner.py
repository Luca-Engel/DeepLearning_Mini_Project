import evaluate
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

def preprocess_function(examples, feature_extractor):
    return feature_extractor(examples["text"], truncation=True)


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


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")
    predictions = np.argmax(eval_pred.predictions, axis=1)

    return {
        **accuracy.compute(predictions=predictions, references=eval_pred.label_ids),
        **recall.compute(predictions=predictions, references=eval_pred.label_ids),
        **precision.compute(predictions=predictions, references=eval_pred.label_ids),
        **f1.compute(predictions=predictions, references=eval_pred.label_ids)
    }
    
    # return f1.compute(predictions=predictions, references=eval_pred.label_ids)