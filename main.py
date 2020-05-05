import json
import numpy as np
import random
import pickle
import tflearn
import tensorflow as tf

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

with open("intents.json") as myfile:
	data = json.load(myfile)
try:
	a
	with open("input_data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except:
	words = []
	labels = []
	docs_x = []
	docs_y = []
	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern) #break the sentences into words
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])
		if intent["tag"] not in labels:
			labels.append(intent["tag"])
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))
	labels = sorted(labels)
	#one hot encoding --- bag of words
	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x,doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:#all
			if w in wrds:#current
				bag.append(1)
			else:
				bag.append(0)

		
		output_row = out_empty[:]     #copy
		output_row[labels.index(docs_y[x])] = 1 

		#print("\n bag: "+str(bag))
		#print("\n row: "+str(output_row))

		training.append(bag)
		output.append(output_row)

	training = np.array(training)
	output = np.array(output)

	with open("input_data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output),f) 
	

tf.reset_default_graph()

network = tflearn.input_data(shape=[None, len(training[0])])

network = tflearn.fully_connected(network,8)
network = tflearn.fully_connected(network,8)

network = tflearn.fully_connected(network,len(output[0]),activation="softmax")
network = tflearn.regression(network)

model = tflearn.DNN(network)

try:
	a
	model.load("chatbot.tflearn")
except:
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("chatbot.tflearn")
	
def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1

	return np.array(bag)

def start_chat():
	print("\n\nBot is ready to talk to you. (type 'quit' to stop) ")
	while True:
		inp = input("You: ")
		if inp.lower() in ["quit","exit"]:
			break

		results = model.predict([bag_of_words(inp,words)])[0] 
		#print(results)
		results_index = np.argmax(results)
		tag = labels[results_index]
		#print(tag)  

		if results[results_index] < 0.8 or len(inp)<2:
			print("Bot: Sorry, I didn't get you. Please try again.\n")
 				
		else:
			for tg in data['intents']:
				if tg['tag'] == tag:
					responses = tg['responses']

			print("Bot: "+random.choice(responses)+"\n")
		
			
start_chat() 