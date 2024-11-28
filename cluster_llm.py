# Import python packages
import pandas as pd
import numpy as np
import random

#from imp import reload

import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import spacy
# Load the English model
nlp = spacy.load("en_core_web_sm")
from collections import Counter
"""
This file contains a class that performs clustering of words and/or sentences using vector embedding and k-means
In addition, the cluster can be named using a call to generative LLM with appropriate prompt.
"""

from itertools import islice

from collections import Counter

import re
from os import environ
import textwrap
import pprint


class WordCluster:
    def __init__(self, word_list, cnt_thresh=20):

        self.word_list = word_list
        self.word_dict = {k:v for k, v in Counter(word_list).items() if v > cnt_thresh}
        self.word_df = pd.DataFrame(columns=["words", "count"], data=list(self.word_dict.items())).sort_values("count", ascending=False).reset_index()
        
        self.labels = None
        self.embeddings = None
        
        self.cluster_name = None
        self.clusters = None


    def plot_word_raw(self, plt_kargs={}, xticks_kwargs={}):
        plt.bar(self.word_df["words"], self.word_df["count"], **plt_kargs)
        plt.xticks(rotation= 90, **xticks_kwargs)
        plt.ylabel("Count")
        plt.xlabel("Words")
        plt.show()

    def wordcloud_raw(self):
        # word clouds
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(self.topic_dict)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    
    def get_embendings(self, model='sentence-transformers/all-MiniLM-L6-v2', llm_context=None):
        # other possible model: "hkunlp/instructor-large"
        
        if llm_context is not None:
            words_to_embedd = [llm_context+w for w in self.word_df.words.to_list()]
        else:
            words_to_embedd = self.word_df.words.to_list()
            
        # Load the pre-trained SBERT model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = model.encode(words_to_embedd)
        
        return self.embeddings
    
    def make_cluster(self, num_clusters=20):
        
        if self.embeddings is None:
            self.get_embendings()
        
        if num_clusters > len(self.word_list):
            raise ValueError("Number of clusters must be smaller than number of words")
            
        # Cluster topics using embedding
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(self.embeddings)
        self.labels = kmeans.labels_
        
        # add labels 
        self.word_df["labels"] = self.labels.astype(str)

        # Print out the words in each cluster
        self.clusters = {i: [] for i in range(num_clusters)}
        for word, label in zip(self.word_df.words.to_list(), self.labels):
            self.clusters[label].append(word.strip())
    
    def name_cluster(self, model="HuggingFaceTB/SmolLM2-1.7B-Instruct", prompt=None, device="cpu", max_items=100):
        
        if not self.clusters:
            raise ValueError("The clusters must first be generated. Call the make_clusters() function")
        
        if prompt is None:
            prompt = "Based on its content, assign one meaningful name to this list. Answer with one word only:"

        tokenizer = AutoTokenizer.from_pretrained(model)
        # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")`
        model = AutoModelForCausalLM.from_pretrained(model).to(device)

        self.cluster_name = {}
        for i, cluster_list in self.clusters.items():
            if len(cluster_list) > 1:
                
                if len(cluster_list) > max_items:
                    shuffled_list = random.sample(cluster_list, max_items)
                else:
                    shuffled_list = cluster_list.copy()
                
                tmp_prompt = f"Here is a list of items: {str(shuffled_list)}. "
                input_text=tokenizer.apply_chat_template([{"role": "user", "content": tmp_prompt+prompt}], tokenize=False)
                
                inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
                outputs = model.generate(inputs, max_new_tokens=10, temperature=0.001, do_sample=True)
                
                response = tokenizer.decode(outputs[0])
                #print(response)
                response = response[response.find("<|im_start|>assistant\n")+len("<|im_start|>assistant\n"):response.rfind("<|im_end|>")]
            else:
                response = cluster_list[0]
            
            self.cluster_name.update({i: response})

    def plot_silhouette(self, k_range=[10, 50]):
        
        if self.embeddings is None:
            self.get_embendings()
            
        if k_range[1] > len(self.word_list):
            raise ValueError("Number of clusters must be smaller than number of words")
                    
        silhouette_scores = []
        k_values = range(k_range[0], k_range[1])  # Test for k=2 to k=20

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.embeddings)
            silhouette_scores.append(silhouette_score(self.embeddings, kmeans.labels_))

        # Plot Silhouette Scores
        plt.figure(figsize=(10, 5))
        plt.plot(k_values, silhouette_scores, marker='o')
        plt.title("Silhouette Analysis")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.show()


    def print_clusters(self):
        # Prints the nicely formatted dictionary
        pprint.pprint(self.clusters)
    
    def plot_clusters(self):
        
        # plot clusterd topics
        tmp = self.word_df.groupby("labels")["count"].sum().reset_index().sort_values("count", ascending=False)
        plt.bar(tmp["labels"], tmp["count"])
        plt.xticks(rotation= 90)
        plt.ylabel("Count")
        plt.xlabel("Word cluster")
        plt.show()
        
        
def get_hf_llm_response(prompt, model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device="cpu"):
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(model).to(device)
                
    input_text=tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
    
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=10, temperature=0.001, do_sample=True)
    
    return tokenizer.decode(outputs[0])


# classical nlp

# Generate n-grams
def generate_ngrams(tokens, n=2):
    ngrams = zip(*(islice(tokens, i, None) for i in range(n)))
    return [" ".join(ngram) for ngram in ngrams]

# Lemmatize, remove stop words, punctuation, and make text lowercase
def preprocess_text(text, thresh_token_len=2):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token)>thresh_token_len]

def analyze_cluster(text):
    
    tokens = preprocess_text(text)
    
    # Unigram frequencies
    unigram_freq = Counter(tokens)
    
    # Bigram frequencies
    bigrams = generate_ngrams(tokens, n=2)
    bigram_freq = Counter(bigrams)
    
    # Trigram frequencies
    trigrams = generate_ngrams(tokens, n=3)
    trigram_freq = Counter(trigrams)
    
    return unigram_freq, bigram_freq, trigram_freq

# Function to plot n-gram frequencies
def plot_ngram_frequencies(ngram_freq, title="N-gram Frequencies", top_n=10):
    # Get the most common n-grams
    most_common = ngram_freq.most_common(top_n)
    ngrams, counts = zip(*most_common)  # Separate n-grams and their counts

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(ngrams, counts)
    plt.xlabel("Frequency")
    plt.ylabel("N-grams")
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.show()
    
