from cursor import VACANCIES
import pandas as pd
from utils import textprocessing
import json

OFFLINE_DATASET_NAME = 'dataset.xlsx'
CONFIG_FILENAME = 'config.json'
DOCUMENTS_COUNT = VACANCIES.count_documents({})

def prep(dataset_name: str):
    """
    :param dataset_name: путь к предварительно собранному .xlsx датасету с HH.ru
    :return: обработанный датасет в виде pd.DataFrame и массив возможных действий DQN-агента
    """
    dataset_as_df = pd.read_excel(dataset_name, index_col=0)
    dataset_as_df = dataset_as_df[~dataset_as_df['description'].duplicated()]
    wrapper = textprocessing.preprocessingWrapper()
    dataset_as_df['stemmed_description'] = wrapper(dataset_as_df, 'description')
    dataset_as_df['industry'] = dataset_as_df['industry'].str.lower().str.replace("'", '').str.replace('[', '') \
    .str.replace(']', '').str.replace('\s*', '', regex=True).str.split(',')
    actions_array = dataset_as_df['industry'][~dataset_as_df['industry'].isna()].explode().unique()
    dataset_as_df = dataset_as_df.loc[:, [
           'id', 'name', 'description', 'stemmed_description', 'industry', 'logo', '@workSchedule', 'compensation'
        ]
    ]
    return dataset_as_df, actions_array

upload_interrupt = False

# проверка наличия записей в базе и их перезапись на новые
if DOCUMENTS_COUNT > 0:
    soft_check = str(input('Предыдущие записи будут удалены. Вы уверены? (y/n)'))
    if soft_check.lower() == 'n':
        upload_interrupt = True
    else:
        VACANCIES.delete_many({})

if not upload_interrupt:
    df, a_arr = prep(OFFLINE_DATASET_NAME)
    try:
        current_config = None

        # обновление массива действий в файле с конфигом
        with open(f'../{CONFIG_FILENAME}', 'r', encoding='utf-8') as f:
            current_config = json.load(f)
        with open(f'../{CONFIG_FILENAME}', 'w', encoding='utf-8') as f:
            current_config['a2idx'] = {idx: action for idx, action in enumerate(a_arr)}
            json.dump(current_config, f, ensure_ascii=False, indent=4)

        # загрузка датасета в базу данных
        VACANCIES.insert_many(df.to_dict(orient='records'))

        print('Датасет успешно загружен')
    except Exception as err:
        print(f'Возникла проблема при загрузке датасета: {err}')
