import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
chat_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

data = pd.read_json('./drive/MyDrive/dataset.json')
labels = data['label'].unique()
contexts = data['context'].unique()

def generate_response(req):
	model = keras.models.load_model('./lstm.keras')
	tokens = Tokenizer.texts_to_sequences([req])
	tokens = pad_sequences(tokens, maxlen = 880)
	prediction = model.predict(np.array(tokens))
	pred = np.argmax(prediction)
	print("It is a question related to " + labels[pred])
	input_ids = tokenizer.encode(req, contexts[pred], padding=True,max_length = 1000)

	sep_index = input_ids.index(tokenizer.sep_token_id)
	num_seg_a = sep_index + 1
	num_seg_b = len(input_ids) - num_seg_a
	segment_ids = [0]*num_seg_a + [1]*num_seg_b

	assert len(segment_ids) == len(input_ids)

	outputs = model(torch.tensor([input_ids]),
								token_type_ids=torch.tensor([segment_ids]),
								return_dict=True)

	start_scores = outputs.start_logits
	end_scores = outputs.end_logits
	answer_start = torch.argmax(start_scores)
	answer_end = torch.argmax(end_scores)

	# Combine the tokens in the answer and print it out.
	answer = ' '.join(tokens[answer_start:answer_end+1])

	print('Answer as tokens: "' + answer + '"')
	return answer