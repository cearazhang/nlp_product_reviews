"""
Ceara Zhang and Jocelyn Ju
DS 3500 / Reusable NLP Library
Created: 19 Feb 2023
Updated: 27 Feb 2023
"""

from collections import Counter, defaultdict
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import math
import itertools
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import os.path
from twilight_exceptions import TextFileError
from pathlib import Path


class TextAnalysis:
    """
    class that loads and preprocesses text files
    to create 3 visualizations using the textual data
    """

    def __init__(self):
        """
        initializes a new text analysis instance with an empty list of
        text files and wc
        """
        self.data = defaultdict(dict)
        self.words = []

    def _preprocess(self, filename):
        """
        private method that removes punctuation and capitalization from
        the text file, calculates wc, and returns a dictionary
        :param filename: (txt) a textfile to preprocess
        :return: a dictionary including file name, sentiment, most common words, sentence length,
                 total wordcount, unique wordcount, words type filtered, and words without stopwords
        """

        # assert and raise custom exceptions
        try:
            assert Path(filename).is_file() is True, "File not found"
            assert type(filename) == str, "Expecting text, received " + str(type(text))
            assert os.path.getsize(filename) > 0, "File is empty"
            assert os.path.splitext(filename)[-1] == '.txt', "Expecting .txt, use custom parser for '" + \
                                                             str(os.path.splitext(filename)[-1]) + "' file"

        except Exception as e:
            raise TextFileError(filename, str(e))

        # read text file
        with open(filename) as f:
            texts = f.read()

        # tokenize sentences to a list of sentences
        sentences = nltk.sent_tokenize(texts)

        # remove unnecessary spaces punctuation, capitalization
        sentences = [re.sub(r'[^\w\s]', '', sent) for sent in sentences]
        sentences = [sent.lower() for sent in sentences]

        # filter out stop words
        filtered_words = TextAnalysis._remove_stopwords(sentences)

        # list of list of words for each sentence
        words = [nltk.word_tokenize(sent) for sent in sentences]

        # determine the sentence count of the file
        sent_count = len(sentences)

        # counter for each word
        common_word_counter = Counter(filtered_words).most_common(10)

        # unique word count
        unique_wc = len(set(filtered_words))

        # total word count, includes stop words
        total_wc = len(texts.split())

        # average sentence length, includes stop words
        av_sent_length = total_wc / sent_count

        sentiment = TextAnalysis._sentiment_analysis(words)[0]

        pos_sent_words = TextAnalysis._sentiment_analysis(words)[1]

        self.data = {'1. file name': filename,
                     '2. sentiment': sentiment,
                     '3. 10 most common words': common_word_counter,
                     '4. average sentence length': av_sent_length,
                     '5. total wordcount': total_wc,
                     '6. unique wordcount': unique_wc,
                     '7. words (adj, adv only)': pos_sent_words,
                     '8. words (w/out stop words)': filtered_words}
        # except Exception as e:
        #     raise TextFileError(text, str(e))

        return self.data

    @staticmethod
    def _remove_stopwords(sent):
        """
        removes stop words from a list of sentences(strings)
        :param: (li) a list of sentences (strings)
        :return: (li) the input with stopwords removed
        """
        filtered_words = []
        for s in sent:
            tokens = nltk.word_tokenize(s)
            tokens = [word for word in tokens if word not in stopwords.words('english')]
            filtered_words.extend(tokens)
        return filtered_words

    @staticmethod
    def _sentiment_analysis(words):
        """
        returns the sentiment scores and list of adjectives
        for a given text
        :param ls_str: (list) list of string
        :return: sentiment analysis
        """
        # add parts of speech tags and combine into one list of words
        pos_sent = [nltk.pos_tag(sent) for sent in words]
        pos_sent = list(itertools.chain.from_iterable(pos_sent))

        # only grabbing adjectives and adverbs
        # return one string of words
        # (tag= JJ, JJR, JJS, RB, RBR, RBS)
        pos_sent = [word for (word, tag) in pos_sent if
                    tag.startswith("JJ")]
        pos_sent_str = ' '.join(pos_sent)

        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(pos_sent_str)
        return sentiment, pos_sent

    def load_text(self, text, label="", parser=None):
        """
        load a text file with the library
        :param: text (txt): a text file to load
        :param: label (str): an optional parameter to label the text
        :param: parser (parser): an optional parameter to pass in a custom parser
        :return: results: parsed text file
        """
        if parser is None:
            # default parsing of standard .txt file
            results = TextAnalysis._preprocess(self, text)

        else:
            # custom parser if passed in
            results = parser(text)

        if label == "":
            label = text

        return results

    def _save_results(self, label, results):
        """
        Integrate parsing results into internal state
        :param: label: unique label for a text file that we parsed
        :param: results: the data extracted from the file as a dictionary attribute-->raw data
        :return: saved results
        """
        for k, v in results.items():
            self.data[k][label] = v

    @staticmethod
    def _stack_columns(df, *cols, vals):
        """
        stacks rows of a dataframe to create binary links of sources and targets
        :param df (df): dataframe
        :param cols (li): columns of interest
        :param vals (int): number of occurrences to be used as a thickness value
        :return: result_df (df): a new dataframe
        """

        # dataframe with the source, target and num column that will be concatted on later
        result_df = pd.DataFrame(columns=['src', 'targ', 'num'])

        # iterate through the list of columns with the case of a value is given or not
        for i in range(0, len(cols) - 1):
            src_col = cols[i]
            targ_col = cols[i + 1]
            if vals:
                stacked_df = df.groupby([src_col, targ_col])[vals].sum().reset_index()
            else:
                stacked_df = df.groupby([src_col, targ_col]).size().reset_index()

        stacked_df.columns = ['src', 'targ', 'num']
        result_df = pd.concat([result_df, stacked_df])

        return result_df

    @staticmethod
    def _code_mapping(df, src, targ):
        """
        create distinct labels for mapping
        :param df: dataframe
        :param src: source column
        :param targ: target column
        :return: dataframe and a list of labels
        """

        # Get distinct labels
        labels = sorted(list(set(list(df[src]) + list(df[targ]))))

        # Get integer codes
        codes = list(range(len(labels)))

        # Create label to code mapping
        lc_map = dict(zip(labels, codes))

        # Substitute names for codes in dataframe
        df = df.replace({src: lc_map, targ: lc_map})

        return df, labels

    @staticmethod
    def get_common_words_from_files(file_list, k):
        """
        make a dataframe of most common words across
        a list of text files
        :param k: number of words to appear on sankey
        :param file_list: list of files to pull words from
        :return: dataframe with columns for file name, words and counts
        """

        # make a list of all words
        # across all given files
        data = []
        all_words = []
        for file in file_list:
            w = file['7. words (adj, adv only)']
            all_words.extend(w)

        # get the word count of all words,
        # and pull the 20 most common
        wc = Counter(all_words)
        most_common = wc.most_common(k)

        # for each file, store the file name,
        # and any common word that they contain
        # along with the count
        for file in file_list:
            file_name = file['1. file name']
            words = file['7. words (adj, adv only)']

            # for each word in most common list,
            # check if word is in
            for word, ct in most_common:
                if word in words:
                    count = words.count(word)
                    data.append([file_name, word, count])

        # create a dataframe with source, target, and value columns
        df = pd.DataFrame(data, columns=['source', 'target', 'values'])
        df[['source', 'target']] = df[['source', 'target']].astype(str)
        return df

    @staticmethod
    def make_sankey(texts, k=20, **kwargs):
        """
        make a sankey diagram
        :param k: (int) number of words to appear on sankey
        :param texts: the list of preprocessed text files
        :param kwargs:
        :return: sankey diagram
        """

        # create a dataframe from the k most common words
        df = TextAnalysis.get_common_words_from_files(texts, k)
        df = TextAnalysis._stack_columns(df, 'source', 'target', vals='values')
        src, targ, vals = 'src', 'targ', 'num'

        # get the labels for diagram
        df, labels = TextAnalysis._code_mapping(df, src, targ)

        # obtain source, target, and values
        link = {'source': df[src], 'target': df[targ], 'value': df[vals]}
        pad = kwargs.get('pad', 50)
        node = {'label': labels, 'pad': pad}

        # create the sankey diagram
        sk = go.Sankey(link=link, node=node)
        fig = go.Figure(sk)

        # set the dimensions for the figure
        width = kwargs.get('width', 800)
        height = kwargs.get('height', 800)
        fig.update_layout(
            autosize=False,
            width=width,
            height=height)

        fig.show()
    
    @staticmethod
    def pie_chart_viz(texts, title=""):
        """
        Creates subplots (pie charts) for each text displaying positive, negative, and neutral sentiment
        :param texts: (list) a list of texts to analyze
        :param title: (str) title for the group of subplots
        :return: a visualization of len(texts) pie charts to compare sentiment scores
        """

        # create the figure
        fig, axs = plt.subplots(1, len(texts), figsize=(11, 5))
        fig.suptitle(title)

        # for each text, take the sentiments and remove the compound
        for i, j in enumerate(texts):
            tempdict = dict(j['2. sentiment'])
            size = []
            comp = tempdict['compound']

            tempdict.pop('compound')

            for x, y in tempdict.items():
                size.append(y)

            # create a pie chart with the sentiments
            axs[i].pie(size)
            axs[i].set_title("Review #" + str(i + 1) + "\n score:" + str(comp), fontsize=10)

        labels = ['negative', 'neutral', 'positive']

        # place the legend for the plots
        fig.legend(labels, loc="upper left")
        plt.show()

    @staticmethod
    def total_viz(texts, title=""):
        """
        create a bar plot with data from all of the text files and a sentiment score line
        :param: texts: (list) a list of texts
        :param: label: (str) an optional label
        :return: visualization: a bar plot displaying each word count overlaid by a line displaying sentiment
                                for each text file
        """

        # initialize empty lists for reviews, values, and sentiment
        reviews = []
        values = []
        sentiment = []

        # for each text, retrieve wordcount and add to appropriate list
        for i, j in enumerate(texts):
            values.append(j['5. total wordcount'])
            reviews.append(str(i + 1))

            tempdict = dict(j['2. sentiment'])
            sentiment.append(tempdict['compound'])

        # create the bar plot
        color_list = ['tomato', 'sandybrown', 'lemonchiffon', 'palegreen', 'mediumseagreen', 'mediumturquoise',
                      'cadetblue', 'slateblue', 'mediumorchid', 'palevioletred']
        label_list = ['Review 1', 'Review 2', 'Review 3', 'Review 4', 'Review 5',
                      'Review 6', 'Review 7', 'Review 8', 'Review 9', 'Review 10']

        plt.bar(reviews, values, color=color_list[:len(reviews)], label=label_list[:len(reviews)])
        plt.legend(loc='upper right')
        plt.xlabel("Review Number")
        plt.ylabel("Total Words (bars)")

        # overlay the sentiment plot
        axis2 = plt.twinx()
        axis2.plot(reviews, sentiment, marker='o', color="k", label="sentiment")
        axis2.set_ylabel("Sentiment (line)")

        # show the plot
        plt.title(title)
        plt.show()


# create an instance of class TextAnalysis
text_analysis = TextAnalysis()
