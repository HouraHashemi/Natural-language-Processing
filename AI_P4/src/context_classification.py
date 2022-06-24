import copy
import sys
import os
import time
import random
import matplotlib.pyplot as plt


class Context_Classification():
    
    def __init__(self):
        # each element is tuple(row,col)
        self.SYMBOLS = '~!@#$%^&*-+={\}()[].,:;/|<>?\'\"_1234567890\n'

        self.dataset = {"pos":"", "neg":""}

        self.negative_words = dict()
        self.positive_words = dict()

        self.negative_probability_one_word = dict()
        self.positive_probability_one_word = dict()

        self.negative_probability_two_words = dict()
        self.positive_probability_two_words = dict()


    def remove_symbols(self,txt):
        for s in self.SYMBOLS:
            txt = txt.replace(s," ")
        txt = txt.replace("  "," ")
        wrd_of_txt = txt.split()
        for t in txt.split():
            if len(t)<=1 :
                wrd_of_txt.remove(t)
        txt = " ".join(wrd_of_txt)
        return txt


    def load_dataset(self,file_name):
        file_type = file_name.split("/")[-1]
        self.file = open(file_name,"r")

        text = "".join([line for line in self.file])
        text = self.remove_symbols(text)

        words_list = list(set(text.split()))
        temp_ds = dict(zip(words_list,[text.count(w) for w in words_list]))
        
        if "pos" in file_type:
            self.positive_words = copy.deepcopy(temp_ds)
            self.dataset["pos"] = text
        elif "neg" in file_type:
            self.negative_words = copy.deepcopy(temp_ds)
            self.dataset["neg"] = text


    def filter_repetitive_words(self):

        pos = copy.deepcopy(self.positive_words)

        high_repetitive_pos = dict(filter(lambda elem: elem[1]>=1000, pos.items())).keys()
        low_repetitive_pos  = dict(filter(lambda elem: elem[1]<=3  , pos.items())).keys()
        repetitive_pos = list(high_repetitive_pos) + list(low_repetitive_pos)
        self.positive_words = dict(filter(lambda elem: elem[0] not in repetitive_pos, pos.items()))

        neg = copy.deepcopy(self.negative_words)

        high_repetitive_neg = dict(filter(lambda elem: elem[1]>=1000, neg.items())).keys()
        low_repetitive_neg  = dict(filter(lambda elem: elem[1]<=3  , neg.items())).keys()
        repetitive_neg = list(high_repetitive_neg) + list(low_repetitive_neg)
        self.negative_words = dict(filter(lambda elem: elem[0] not in repetitive_neg, neg.items()))


    def calculate_probability_one_word(self,wi):
        # p(wi) = count(wi)/M 
        
        M_pos = sum(self.positive_words.values())
        self.positive_probability_one_word[wi] = round(self.dataset["pos"].count(wi)/M_pos, 4)        
            
        M_neg = sum(self.negative_words.values())
        self.negative_probability_one_word[wi] = round(self.dataset["neg"].count(wi)/M_neg, 4)
            

    def calculate_probability_two_words(self,w):
        # P(w1|w0) = count(w0w1)/count(w0) 
        cw0w1_pos = self.dataset["pos"].count(str(w[0] + " " + w[1]))
        if cw0w1_pos == 0 :
            # one of two words doesnt exist
            pw1w0_pos = .0  
        else:
            pw1w0_pos = cw0w1_pos/self.dataset["pos"].count(str(w[0]))
        self.positive_probability_two_words[w] = round(pw1w0_pos)
        # negative form
        cw0w1_neg = self.dataset["neg"].count(str(w[0] + " " + w[1]))
        if cw0w1_neg == 0:
            pw1w0_neg = .0
        else:
            pw1w0_neg = cw0w1_neg/self.dataset["neg"].count(str(w[0]))
        self.negative_probability_two_words[w] = round(pw1w0_neg)

    
    def calculate_total_probability(self,bigram_list, words_list ,one_word_dict, two_words_dict):
        # P(w1w2w3 ... wn) = P(w1).|-|P(wi|wi-1)
        # P(wi|wi-1) = landa1.P(wi|wi-1) + landa2.P(wi) + landa3.e
        # landa1 + landa2 + landa3 = 1
        probability = one_word_dict[words_list[0]]
        if probability == 0:
            probability = 1
        for b in bigram_list:
            landa_x1 = 0.1
            landa_x2 = 0.4
            landa_x3 = 0.5
            # landa_x1 = (random.choices(range(1,50))[0])/100
            # landa_x2 = (random.choices(range(1,50))[0])/100
            # landa_x3 = 1 - (landa_x1 + landa_x2)
            x1 = 0.1
            x1 = (random.choices(range(1,100))[0])/100
            x2 = one_word_dict[b[0]] # get P(wi) 
            x3 = two_words_dict[b]   # get p(wi|wi-1)
            pw1w0 = landa_x3*(x3) + landa_x2*(x2) + landa_x1*(x1)

            probability = pw1w0
        return probability

      
    def classification_by_bigram(self,sentence):

        sentence = self.remove_symbols(sentence)             
        # words  : ["", "", "", ...]
        # bigram : [(wi,wi-1), (w1,w0), (am,i), ...] 
        words = [" "] + sentence.split() + [" "]
        bigram = [ b for b in zip(words[1:], words[:-1])]
        
        # calculate P(wi)
        for w in words:
            self.calculate_probability_one_word(w)
        # calculate P(wi|wi-1)
        for b in bigram:
            self.calculate_probability_two_words(b)

        # calculate posetive probability 
        pp = self.calculate_total_probability(bigram,words,self.positive_probability_one_word,\
           self.positive_probability_two_words)
        # calculate negative probability 
        np = self.calculate_total_probability(bigram,words,self.negative_probability_one_word,\
           self.negative_probability_two_words)
        
        # print("pp:{}, np:{}".format(pp,np))

        if np<pp:
            # # opinion is positive
            print("[✔️  ] NOT FILTER THIS")
            return True
        elif np>pp:
            # # opinion is negative
            print("[❌ ] FILTER THIS")
            return False
        else:
            print("[❔] UNKNOWN")
            return None


    def classification_by_unigram(self,sentence):

        sentence = self.remove_symbols(sentence)
        words = sentence.split()  
        # calculate P(wi)
        for w in words:
            self.calculate_probability_one_word(w)

        # calculate posetive probability 
        pp = 1
        for ppow in self.positive_probability_one_word.items():
            pp = pp*ppow[1]
        # calculate negative probability 
        np = 1
        for npow in self.negative_probability_one_word.items():
            np = np*npow[1]

        if np<pp:
            # # opinion is positive
            print("[✔️  ] NOT FILTER THIS")
            return True
        elif np>pp:
            # # opinion is negative
            print("[❌ ] FILTER THIS")
            return False
        else:
            print("[❔ ] UNKNOWN")
            return None

       
#---------------------------------------------------------
# END OF CLASS
#---------------------------------------------------------


if __name__ == '__main__':
    
    pos_dataset_path = "dataset/rt-polarity-pos.txt"
    neg_dataset_path = "dataset/rt-polarity-neg.txt"
    
    cc = Context_Classification()


    print("=================================================")
    print("------ 📑 👓 LEARNING MODELS STARTED 👓 📑 ------")
    print("=================================================")
    
    model_type = input("[🔻 ] FILTER MODEL [B: BIGRAM | U: UNIGRAM]: ")
    if (model_type not in "bB") and (model_type not in "uU"):
        print("[⚠️  ] MODEL TYPE NOT DEFINED!")
        sys.exit()

    print("=================================================")
    print("⏳ PLEASE WAITE ... ", end=" ")

    cc.load_dataset(pos_dataset_path)
    cc.load_dataset(neg_dataset_path)
    cc.filter_repetitive_words()
    os.system('cls' if os.name == 'nt' else 'clear')

    print("==================================================")
    print("------ ⚙️  💡 LEARNING MODELS COMPLETED 💡⚙️  ------")
    print("==================================================")
    term = ""          

    if model_type in "Bb":
        print(" ⚪ FILTERED BY BIGRAM")
        print("=================================================")
    else:
        print(" ⚫ FILTERED BY UNIGRAM")
        print("=================================================")
    

    while(True):
        term = input("> ")
        if term == '!q':
            break    
        term = cc.remove_symbols(term)
        if model_type in "Bb":
            cc.classification_by_bigram(term)
        else:
            cc.classification_by_unigram(term)
        print("-------------------------------------------------")

        




