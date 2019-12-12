import json

PAD_token = 0  
SOS_token = 1 
EOS_token = 2  

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)
    def save(self):
        path = "Data/save/Voc_mode/"
        with open(f"{path}word2index.txt", 'w') as fh:
            json.dump(self.word2index, fh)
        with open(f"{path}word2count.txt", 'w') as fh:
            json.dump(self.word2count, fh)
        with open(f"{path}index2word.txt", 'w') as fh:
            json.dump(self.index2word, fh)
        
        
    def load(self):
        path = "Data/save/Voc_mode/"
        with open(f"{path}word2index.txt", 'r') as fh:
            self.word2index = json.load(fh)
        with open(f"{path}word2count.txt", 'r') as fh:
            self.word2count = json.load(fh)
        with open(f"{path}index2word.txt", 'r') as fh:
            self.index2word = json.load(fh)
        self.num_words = len(self.index2word)
