import certifi
from pymongo import MongoClient
from dotenv import load_dotenv
from os import getenv
import pandas as pd
from utils.dataframe_ops import create_onehot_masked_df

load_dotenv()
ca = certifi.where()

DB_URI = getenv('URI')
ClientForDB = MongoClient(DB_URI, tlsCAFile=ca)
COLLECTION = ClientForDB['ml']
RECORDS = COLLECTION['dqn_records'] # записи о состояниях среды
VACANCIES = COLLECTION['dataset'] # база с датасетом загруженным с помощью скрипта в ./upload_dataset.py

def load_stemmed_descriptions() -> pd.DataFrame:
    """
    :return: Датафрейм из ID, названия, описания, стемматизированного описания и логотипа с вакансии с
    one-hot энкодированными столбцами для индустрий
    """
    result = list(VACANCIES.find({}, {'id': 1, 'description': 1, 'stemmed_description': 1, 'name': 1, 'logo': 1, 'industry': 1}))
    result_df = pd.DataFrame(list(
        map(lambda record:
            {'id': record['id'], 'name': record['name'], 'desc': ' '.join(record['stemmed_description']), 'industries': record['industry'],
             'desc_full': record['description'], 'logo': record['logo']},
            result)
    ))
    one_hot_industries = create_onehot_masked_df(result_df['industries'])
    return pd.concat([result_df[['id', 'name', 'desc', 'desc_full', 'logo']], one_hot_industries], axis=1)

