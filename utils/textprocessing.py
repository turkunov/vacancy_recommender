import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
import re

class preprocessingWrapper:

  """
  Основной класс для препроцессинга текста и приведения его к
  стеммизированной форме с помощью SnowballStemmer. 
  Список из уникальных стоп слов также был немного расширен популярными словами
  в описаниях вакансий.
  """

  def __init__(self):
    nltk.download('punkt')
    nltk.download('stopwords')

    self.stemmer = SnowballStemmer('russian')
    self.stop_words = set(
      stopwords.words('russian') + stopwords.words('english')
    ).union({
        'компания', 'работа', 'работы', 'работу',
        'ищем', 'искать', 'мы', 'наш', 'наша', 'наши', 'ооо'
    })

  def rem_stopwords(self, tokens):
    return list(filter(
      lambda token: token or len(token) < 1 not in self.stop_words, tokens
    ))

  def tokenize(self, token_string):
    return [w for s in sent_tokenize(token_string) for w in word_tokenize(s)]

  def stem(self, tokens):
    return [self.stemmer.stem(token) for token in tokens]

  def __call__(
    self, df, column_for_cleaning=None
  ):
    """ 
      Этот основной метод позволяет 'обернуть' датафрейм и очистить столбец 
      
      df: pd.DataFrame
        - Датафрейм (или строка) для очистки
      column_for_cleaning: str [optional]
        - Столбец для очистки
    """

    if isinstance(df, pd.DataFrame):
      column = df[column_for_cleaning]
      column = column.str.lower()

      # удаление и замена спец. символов
      column = column.str.replace('[^0-9A-Za-zА-Яа-я.\s]+', '', regex=True)
      column = column.str.replace('quot','"')

      # токенизация, удаление стоп-слов и стемматизация
      column = column.apply(self.tokenize)
      column = column.apply(self.rem_stopwords)
      column = column.apply(self.stem)

      return column
    else:
      df = df.lower()
      df = re.sub('[^0-9A-Za-zА-Яа-я.\s]+', '',df)
      df = self.tokenize(df) # токенизация
      df = self.rem_stopwords(df)
      df = self.stem(df)
      return ' '.join(df)


