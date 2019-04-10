import MeCab
import neologdn
import mojimoji
import numpy as np



def  normalize_text(text):
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
