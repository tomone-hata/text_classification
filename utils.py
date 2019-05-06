import MeCab
import neologdn
import mojimoji
import numpy as np
import re
import copy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def normalize_text(text):
    result = mojimoji.zen_to_han(text, kana=False)
    result = neologdn.normalize(result)
    return result


def text_to_words(text):
    m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    m.parse('')
    #neologdnにより正規化処理をする。
    text = normalize_text(text)
    m_text = m.parse(text)
    basic_words = []
    #mecabの出力結果を単語ごとにリスト化
    m_text = m_text.split('\n')
    for row in m_text:
        #Tab区切りで形態素、その品詞等の内容と分かれているので単語部のみ取得
        word = row.split("\t")[0]
        #最終行はEOS
        if word == 'EOS':
            break
        else:
            pos = row.split('\t')[1]
            slice_ = pos.split(',')
            #品詞を取得する
            parts = slice_[0]
            if parts == '記号':
                if word != '。':
                    continue

                #読点のみ残す
                basic_words.append(word)
            #活用語の場合は活用指定ない原型を取得する。
            elif slice_[0] in ('形容詞', '動詞'):
                    basic_words.append(slice_[-3])

            #活用しない語についてはそのままの語を取得する
            elif slice_[0] in ('名詞', '副詞'):
                basic_words.append(word)

    basic_words = ' '.join(basic_words)
    return basic_words


def label_to_one_hot(label_size, label):
    vector = np.zeros((label_size),dtype=int)
    vector[label] = 1
    return vector


class ComputeTfidf(object):
    def __init__(self, corpus):
        self.cv = CountVectorizer()
        self.ttf = TfidfTransformer()
        self.corpus = corpus
        self.tf, self.tfidf = None, None


    def compute_tf(self):
        self.tf = self.cv.fit_transform(self.corpus)
        return self.tf


    def compute_tfidf(self):
        self.tfidf = self.ttf.fit_transform(self.tf)
        return self.tfidf


class MeasurementStatistics(object):
    def __init__(self, text_list):
        self.text_list = text_list
        #self.label_list = [x[0] for x in text_list]
        #self.copus = [x[1] for x in text_list]
        self.count_list = None

    def word_count(self, split_=' '):
        self.count_list = np.array([[x[0], len(re.split(split_, x[1]))] for x in self.text_list])
        return self.count_list


    def percent_point(self):
        result_dict = {}
        words_count = np.array(self.count_list[:,1], dtype='int')
        result_dict['sample_num'] = self.count_list.shape[0]
        print(words_count)
        result_dict['min_len'] = np.min(words_count, axis=0)
        result_dict['1Q_len'], result_dict['2Q_len'], result_dict['3Q_len'] = \
                                          np.percentile(words_count,[25, 50, 75])
        result_dict['max_len'] = np.max(words_count, axis=0)
        result_dict['average'] = np.mean(words_count, axis=0)
        return result_dict


class PreprocessCNN(object):
    def __init__(self, train, test):
        self.x_train_in = [x[1] for x in train]
        self.y_train_in = [x[0] for x in train]
        self.x_test_in = [x[1] for x in test]
        self.y_test_in = [x[0] for x in test]
        self.x_train, self.y_train = [], []
        self.x_test, self.y_test = [], []


    @staticmethod
    def _increasing_texts(word_list, padding_size, increasing_min):
        words = []
        return_size = (len(word_list) // padding_size)
        for idx in range(return_size):
            start_idx = padding_size * idx
            end_idx = start_idx + padding_size
            words.append(copy.copy(word_list[start_idx:end_idx]))
        else:
            start_idx = padding_size * return_size
            words.append(copy.copy(word_list[start_idx:]))

        return words


    def data_preprocessing(self, max_words=0, sep=' ', padding_size=4096, increasing_mode=False, increasing_min=2048):
        vocab_size = max_words
        if increasing_mode:
            assert padding_size >= increasing_min, 'Invalid padding_size and increasing_min.'
            print('Start to increase train data.')
            for text, label in zip(self.x_train_in, self.y_train_in):
                text = re.split(sep, text)
                text = [x.strip() for x in text]
                text_size = len(text)
                padding_diff = padding_size - text_size
                text = self._increasing_texts(text, padding_size, increasing_min) \
                       if padding_diff < 0 else [text]
                text = [' '.join(x) for x in text]
                self.x_train.extend(text)
                self.y_train.extend([label for _ in range(len(text))])

            print('Finish to increase train data.')
        else:
            self.x_train = self.x_train_in
            self.y_train = self.y_train_in

        self.x_test = self.x_test_in
        self.y_test = self.y_test_in
        if vocab_size <= 0:
            print('Invalid num_words. Compute vocab size with x_train.')
            corpus = self.x_train
            ct = ComputeTfidf(corpus)
            ct.compute_tf()
            vocab_size = ct.compute_tfidf().shape[1]
            print('Finish to compute vocab size.')

        print('Vocab size: %i' % vocab_size)
        print('Start to texts to sequences')
        print('Train to tokenize.')
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(self.x_train)
        print('Finish to train to tokenize.')
        print('Start to texts to sequences.')
        train_idx = tokenizer.texts_to_sequences(self.x_train)
        test_idx = tokenizer.texts_to_sequences(self.x_test)
        self.x_train = pad_sequences(train_idx, maxlen=padding_size, padding='post', truncating='post')
        self.x_test = pad_sequences(test_idx, maxlen=padding_size, padding='post', truncating='post')
        print('Finish to texts to sequences.')
        return self.x_train, self.y_train, self.x_test, self.y_test
