import numpy as np
import pandas as pd

def words2idxmapping(words_array, word2idx_dict):
  return [word2idx_dict[word] for word in words_array]

def create_onehot_masked_df(column_of_seqs, list_is_string=False):
    """
    Функция, получающая на вход столбец из списков column_of_seqs и
    флаг, отмечающий являются ли записи списками или это строка со списком. Возвращает датаферейм
    из onehot-encoded столбцов уникальных значений, встречающихся во вложенных списках
    """
    column_of_seqs.fillna("", inplace=True)
    if list_is_string:
        column_of_seqs = column_of_seqs.str.replace("'", '').str.replace('[', '') \
            .str.replace(']', '').str.replace('\s*', '', regex=True).str.split(',')
    unique_names = column_of_seqs.explode().unique()
    idx2word = {idx: name for idx, name in enumerate(unique_names)}
    word2idx = {name: idx for idx, name in idx2word.items()}
    seqs2idx = column_of_seqs.apply(
        lambda words: words2idxmapping(words, word2idx)
    )
    onehot_mask = np.zeros((column_of_seqs.shape[0], len(unique_names)))
    for indx, namesmapping in enumerate(seqs2idx.values):
        onehot_mask[indx, namesmapping] = 1

    return pd.DataFrame(onehot_mask, columns=unique_names)