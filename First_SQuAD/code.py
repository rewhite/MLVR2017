import json
import nltk
import itertools
import sys,os
sys.path.append(os.pardir)
from First_GloVe.code import loadGloveModel

print("Started loading SQuAD")
with open("dev-v1.1.json") as dataFile:
    data = json.load(dataFile)
print("Finished loading SQuAD")

context = [data["data"][0]["paragraphs"][0]["context"]]
sentences = itertools.chain(*[nltk.sent_tokenize(cont) for cont in context]) #[]
sentences = ["%s %s %s" %("SENTENCE_START", sent, "SENTENCE_END") for sent in sentences]
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

model = loadGloveModel()

def decode():
    for w in tokenized_sentences[0]:
        if w in model:
            print(w)
            print(model[w])
    print("")
    for w in tokenized_sentences[0]:
        print(w)


    # print("Vector")
    # print





# ['Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.', 'The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.', "The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.", 'As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.']
