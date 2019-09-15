import re
import glob
import nltk, pylab, math
from scipy import stats
from nltk.util import ngrams
from nltk.corpus import wordnet as wn, stopwords
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from itertools import chain
import matplotlib.pyplot as plt

class NLP:
    def __init__(self, path, ismerge=0):
        self.path = path
        self.ismerge = ismerge
        self.data = self.getData()
        self.token = self.tokenizeData()
        self.freqMap = None
        self.ambi = None
        self.posTag = None

    def mergeFiles(self):
        read_files = glob.glob(self.path+"*.txt")
        with open("merge.txt", "wb") as outfile:
            for f in read_files:
                    with open(f, "rb") as infile:
                        outfile.write(infile.read())

    def getData(self):
        if self.ismerge == 1:
            self.mergeFiles()
        with open("merge.txt", 'r') as file:
            text = file.read()
            #text = text.replace("@ @ @ @ @ @ @ @ @ @", '').replace("<p>", '').replace(',', '').replace(r'[0-9]+', '')
            text = re.sub("@ @ @ @ @ @ @ @ @ @", '', text)
            text = re.sub("<p>", '', text)
            text = re.sub("[0-9]", '', text)
            text = re.sub(",", '', text)
            text = re.sub("``", '', text)
            #text = re.findall(r'\w+', text)
            file.close()
            text = text.split('.')
        return text[0:20000]


    def tokenizeData(self):
        token = []
        for sentence in self.data:
            token += nltk.word_tokenize(sentence)
        stop_words = set(stopwords.words('english')) 
        filtered_token = []
        for w in token:
            if w not in stop_words or w != '`':
                filtered_token.append(w)
        return filtered_token

    def zeffLaw(self):
        tokens = self.token
        tokens = [token.lower() for token in tokens if len(token) > 1]
        fdist = nltk.FreqDist(tokens)
        words = fdist.most_common()
        self.freqMap = words
        x = [math.log10(i[1]) for i in words]
        y = [math.log10(i) for i in range(1, len(x))]
        x.pop()
        (m, b) = pylab.polyfit(x, y, 1)
        yp = pylab.polyval([m, b], x)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        pylab.plot(x, yp)
        pylab.scatter(x, y)
        pylab.ylim([min(y), max(y)])
        pylab.xlim([min(x), max(x)])
        pylab.grid(True)
        pylab.ylabel('Counts of words')
        pylab.xlabel('Ranks of words')
        pylab.figtext(.3, .5, "R^2 = 0.7992869")
        pylab.savefig(r'./Zipf.pdf')
    
    def extract_ngrams(self, num):
        print("Length of Vocabulary:", len(self.token))
        n_grams = ngrams(self.token, num)
        return [ ' '.join(grams) for grams in n_grams]

    def pos(self):
        self.posTag = nltk.pos_tag(self.token)

    def getAmbiWords(self, num):
        ambi = []
        for word in self.freqMap:
            numSyms = len(wn.synsets(word[0]))
            if(numSyms != 0):
                ambi.append((word[0], numSyms, word[1], numSyms*word[1]))
        def sortForth(val):
            return val[3]
        ambi.sort(key=sortForth, reverse=1)
        self.ambi = ambi[0:num]


    def disambigious(self):
        for am in self.ambi:
            context = []
            for sentence in self.data:
                if am[0] in sentence:
                    context += sentence
            sense = self.lesk(context, am[0])
            try:
                print("Word", am)
                print("Sense:", sense)
                print("Defination:", sense.definition())
            except Exception as err:
                print(err)

    def lesk(self, context_sentence, ambiguous_word, pos=None, stem=True, hyperhypo=True):
        max_overlaps = 0; lesk_sense = None
        ps = PorterStemmer()
        for ss in wn.synsets(ambiguous_word):
            # If POS is specified.
            if pos and ss.pos is not pos:
                continue

            lesk_dictionary = []

            # Includes definition.
            lesk_dictionary+= ss.definition().split()
            # Includes lemma_names.
            lesk_dictionary+= ss.lemma_names()

            # Optional: includes lemma_names of hypernyms and hyponyms.
            if hyperhypo == True:
                lesk_dictionary+= list(chain(*[i.lemma_names() for i in ss.hypernyms()+ss.hyponyms()]))       

            if stem == True: # Matching exact words causes sparsity, so lets match stems.
                lesk_dictionary = [ps.stem(i) for i in lesk_dictionary]
                context_sentence = [ps.stem(i) for i in context_sentence] 

            overlaps = set(lesk_dictionary).intersection(context_sentence)

            if len(overlaps) > max_overlaps:
                lesk_sense = ss
                max_overlaps = len(overlaps)
        return lesk_sense

    def vocCount(self):
        voc = {}
        for pos in self.posTag:
            if pos[1] in voc:
                voc[pos[1]] += 1
            else:
                voc[pos[1]] = 1 
        plt.clf()
        plt.bar(range(len(voc)), list(voc.values()), align='center')
        plt.xticks(range(len(voc)), list(voc.keys()), rotation=90, fontsize=6)
        plt.savefig('voc.pdf')

if __name__ == "__main__":
	n = NLP(path = '/media/ghost/New Volume/dataset/coco/text/', ismerge = 0)
	n.zeffLaw()
	n.extract_ngrams(1)
	n.pos()
	print(n.vocCount())
	n.getAmbiWords(50)
	#for word in n.ambi:
	#	print(word[0])
	n.disambigious()