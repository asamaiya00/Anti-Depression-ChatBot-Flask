from flask import Flask,render_template,request,redirect,url_for
import tflearn
import numpy as np
import pickle,random
import json

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

app = Flask(__name__)

with open("input_data.pickle", "rb") as f:
	words, labels, training, output = pickle.load(f)

with open("intents.json") as myfile:
	data = json.load(myfile)

#tf.reset_default_graph()

network = tflearn.input_data(shape=[None, len(training[0])])

network = tflearn.fully_connected(network,8)
network = tflearn.fully_connected(network,8)

network = tflearn.fully_connected(network,len(output[0]),activation="softmax")
network = tflearn.regression(network)

model = tflearn.DNN(network)

model.load("chatbot.tflearn")

chats=[]
@app.route("/") #home
def hello():
	return render_template("chat_bot.html",type="start to type")

@app.route("/start",methods=['POST','GET'])
def start():
	inp = [str(x) for x in request.form.values()]
	print(inp[0])
	#return render_template('chat_bot.html',result=inp[0])
	results = model.predict([bag_of_words(inp[0],words)])[0] 
	print(results)
	results_index = np.argmax(results)
	tag = labels[results_index]
	print(tag)  

	if results[results_index] < 0.8 or len(inp[0])<2:
		result ="Sorry, I didn't get you. Please try again."
				
	else:
		for tg in data['intents']:
			if tg['tag'] == tag:
				responses = tg['responses']

		result=""+random.choice(responses)
	chats.append("You: " + inp[0])
	chats.append(result)
	return render_template('chat_bot.html',chats=chats[::-1],type="")

	
def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1

	return np.array(bag)

			
# start() 
if __name__=="__main__":
	app.run(debug=True)

