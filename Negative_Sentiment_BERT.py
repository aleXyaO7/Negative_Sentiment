from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

model = TFBertForSequenceClassification.from_pretrained("Negative_Sentiment_BERT")
tokenizer = BertTokenizer.from_pretrained("Negative_Sentiment_Tokenizer")

def predict(sentence):
    tf_batch = tokenizer(sentence, max_length = 512, padding = True, truncation = True, return_tensors = 'tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    return label

sentence = [input('Enter a sentence:')]
print(predict(sentence))