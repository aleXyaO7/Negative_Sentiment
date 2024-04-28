from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

label, text = 'label', 'sentence'
df = pd.read_csv('dataset.csv')[[text, label]]
print(df.head)
input()

df = df.astype({label: int})

print(df.shape)

df[label].value_counts(normalize = True)
train_text, temp_text, train_labels, temp_labels = train_test_split(df[text], df[label], random_state=2, test_size=0.05, stratify=df[label])

train = pd.concat([train_text, train_labels], axis=1)
train.columns = ['DATA_COLUMN', 'LABEL_COLUMN']

test = pd.concat([temp_text, temp_labels], axis=1)
test.columns = ['DATA_COLUMN', 'LABEL_COLUMN']

from sklearn.utils.class_weight import compute_class_weight

class_wts = compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(train['LABEL_COLUMN']),
    y = train_labels)

print(class_wts)

weights= torch.tensor(class_wts,dtype=torch.float)

cross_entropy = nn.NLLLoss(weight=weights)
epochs = 1

InputExample(guid=None, text_a = "Hello, world", text_b = None, label = 1)

def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN):
    train_InputExamples = train.apply(lambda x: InputExample(guid=None,
                                                            text_a = x[DATA_COLUMN],
                                                            text_b = None,
                                                            label = x[LABEL_COLUMN]), axis = 1)

    validation_InputExamples = test.apply(lambda x: InputExample(guid=None, 
                                                            text_a = x[DATA_COLUMN],
                                                            text_b = None,
                                                            label = x[LABEL_COLUMN]), axis = 1)

    return train_InputExamples, validation_InputExamples

train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, 'DATA_COLUMN', 'LABEL_COLUMN')

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

DATA_COLUMN = 'DATA_COLUMN'
LABEL_COLUMN = 'LABEL_COLUMN'

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs=1, validation_data=validation_data)

from sklearn.metrics import classification_report
preds=model.predict(validation_data)
preds=np.argmax(preds.logits, axis=1)
print(classification_report(temp_labels, preds))

pd.crosstab(temp_labels, preds)

from sklearn.metrics import accuracy_score
accuracy_score(temp_labels, preds)

model.save_pretrained('Negative_Sentiment_BERT')
tokenizer.save_pretrained('Negative_Sentiment_Tokenizer')