from email.mime import audio
import sounddevice as sd
import speech_recognition as sr
import pydub
import spacy
import pytextrank
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from scipy.io import wavfile
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from google.cloud import speech
from transformers import pipeline

'''
def summarize(text,per):
    print(text)
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    print(final_summary)
    summary=''.join(final_summary)

    print(summary)

'''



class summary():
    '''
    @staticmethod
    def speechToText(self):
        pydub.AudioSegment.ffmpeg = "/absolute/path/to/ffmpeg"

        text_file = open("deneme.txt","w")
        r = sr.Recognizer()#tanıma işi burada olacak

        print(sd.query_devices()) # bütün ses i/o cihazlarını listeliyor

        sd.default.device[0] = 2 #hoparlrü seçip default ayarlıyrıuz
        fs = 44100 #frekans
        length = 20 #süre
        recording = sd.rec(frames=fs * length, samplerate=fs, blocking=True, channels=1) #reocr ediyr
        sd.wait()

        wavfile.write('test.wav',fs,recording) # recordu verilen isimle aynı dizine kayıt ediyor
        sound = pydub.AudioSegment.from_wav("test.wav").export("test1.wav", format="wav") #mp3 çeviriyor
        file = sr.AudioFile('test1.wav')#file ptr içine ses doyası atılıyor

        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        
        with file as source:

            #r.adjust_for_ambient_noise(source) #??
            r.energy_threshold=50
            r.dynamic_energy_threshold=False
            audio = r.record(source) # ses dosyasını kayıt ediyoruz
        result = r.recognize_google(audio)#kaydı google apiye yollama
        #result = client.recognize(config=config, audio=audio)
        text_file.write(result)#dosyaya yazma

        text_file.close()
'''
    def dinleAmaBart(self):
        pydub.AudioSegment.ffmpeg = "/absolute/path/to/ffmpeg"

        text_file = open("deneme.txt","w")
        r = sr.Recognizer()#tanıma işi burada olacak

        print(sd.query_devices()) # bütün ses i/o cihazlarını listeliyor

        sd.default.device[0] = 2 #hoparlrü seçip default ayarlıyrıuz
        fs = 44100 #frekans
        length = 20 #süre
        recording = sd.rec(frames=fs * length, samplerate=fs, blocking=True, channels=1) #reocr ediyr
        sd.wait()

        wavfile.write('test.wav',fs,recording) # recordu verilen isimle aynı dizine kayıt ediyor
        sound = pydub.AudioSegment.from_wav("test.wav").export("test1.wav", format="wav") #mp3 çeviriyor
        file = sr.AudioFile('test1.wav')#file ptr içine ses doyası atılıyor

        '''
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        '''
        with file as source:

            #r.adjust_for_ambient_noise(source) #??
            r.energy_threshold=50
            r.dynamic_energy_threshold=False
            audio = r.record(source) # ses dosyasını kayıt ediyoruz
        result = r.recognize_google(audio)#kaydı google apiye yollama
        #result = client.recognize(config=config, audio=audio)
        text_file.write(result)#dosyaya yazma

        text_file.close()
        summarizer = pipeline(result, model="facebook/bart-large-cnn")
        print(summarizer(result, max_length=130, min_length=30, do_sample=False))

    def dinle(self):
        pydub.AudioSegment.ffmpeg = "/absolute/path/to/ffmpeg"

        text_file = open("deneme.txt","w")
        r = sr.Recognizer()#tanıma işi burada olacak

        print(sd.query_devices()) # bütün ses i/o cihazlarını listeliyor

        sd.default.device[0] = 2 #hoparlrü seçip default ayarlıyrıuz
        fs = 44100 #frekans
        length = 20 #süre
        recording = sd.rec(frames=fs * length, samplerate=fs, blocking=True, channels=1) #reocr ediyr
        sd.wait()

        wavfile.write('test.wav',fs,recording) # recordu verilen isimle aynı dizine kayıt ediyor
        sound = pydub.AudioSegment.from_wav("test.wav").export("test1.wav", format="wav") #mp3 çeviriyor
        file = sr.AudioFile('test1.wav')#file ptr içine ses doyası atılıyor

        '''
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        '''
        with file as source:

            #r.adjust_for_ambient_noise(source) #??
            r.energy_threshold=50
            r.dynamic_energy_threshold=False
            audio = r.record(source) # ses dosyasını kayıt ediyoruz
        result = r.recognize_google(audio)#kaydı google apiye yollama
        #result = client.recognize(config=config, audio=audio)
        text_file.write(result)#dosyaya yazma

        text_file.close()
        # ``

        #aşağısı textrank 

        #text = "Billy always listens to his mother. He always does what she says. If his mother says, Brush your teeth, Billy brushes his teeth. If his mother says, Go to bed, Billy goes to bed. Billy is a very good boy. A good boy listens to his mother. His mother doesn't have to ask him again. She asks him to do something one time, and she doesn't ask again. Billy is a good boy. He does what his mother asks the first time. She doesn't have to ask again. She tells Billy, You are my best child. Of course Billy is her best child. Billy is her only child."
        text_file = open("deneme.txt","r")
        text = text_file.read()
        print(text)
        text_file.close()
        # load a spaCy model, depending on language, scale, etc.
        nlp = spacy.load("en_core_web_sm")

        # add PyTextRank to the spaCy pipeline
        nlp.add_pipe("textrank")
        doc = nlp(text)
        print(doc.text)
        # examine the top-ranked phrases in the document
        for token in doc:
            print(token.text, token.pos, token.dep)

        print("----------------------------------")


        for phrase in doc._.phrases:
            print(phrase.text)
            print(phrase.rank, phrase.count)
            print(phrase.chunks)

        print("----------------------------------")

        #summarize(text, 0.05)


        def read_article(file_name):
            file = open(file_name, "r")
            filedata = file.readlines()
            article = filedata[0].split(". ")
            sentences = []

            for sentence in article:
                print(sentence)
                sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
            sentences.pop() 
            
            return sentences

        def sentence_similarity(sent1, sent2, stopwords=None):
            if stopwords is None:
                stopwords = []
        
            sent1 = [w.lower() for w in sent1]
            sent2 = [w.lower() for w in sent2]
        
            all_words = list(set(sent1 + sent2))
        
            vector1 = [0] * len(all_words)
            vector2 = [0] * len(all_words)
        
            # build the vector for the first sentence
            for w in sent1:
                if w in stopwords:
                    continue
                vector1[all_words.index(w)] += 1
        
            # build the vector for the second sentence
            for w in sent2:
                if w in stopwords:
                    continue
                vector2[all_words.index(w)] += 1
        
            return 1 - cosine_distance(vector1, vector2)
        
        def build_similarity_matrix(sentences, stop_words):
            # Create an empty similarity matrix
            similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
            for idx1 in range(len(sentences)):
                for idx2 in range(len(sentences)):
                    if idx1 == idx2: #ignore if both are same sentences
                        continue 
                    similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

            return similarity_matrix


        def generate_summary(file_name, top_n):
            stop_words = set(stopwords.words('english'))
            summarize_text = []

            # Step 1 - Read text anc split it
            sentences =  read_article(file_name)

            # Step 2 - Generate Similary Martix across sentences
            sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

            # Step 3 - Rank sentences in similarity martix
            sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
            scores = nx.pagerank(sentence_similarity_graph)

            # Step 4 - Sort the rank and pick top sentences
            ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
            print("Indexes of top ranked_sentence order are ", ranked_sentence)    

            for i in range(top_n):
                summarize_text.append(" ".join(ranked_sentence[i][1]))

            # Step 5 - Offcourse, output the summarize texr
            print("Summarize Text: \n", ". ".join(summarize_text))

        # let's begin
        generate_summary( "deneme.txt", 1)