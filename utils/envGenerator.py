from utils import textprocessing
import pandas as pd
import numpy as np

class envGenerator:
    
    """
    Класс для взаимодействия со средой-датасетом. Используется только 
    для тестирования работы DQN-агента. Рандомно генирирует массивы
    s, a, r. Записи состояний среды хранятся в атрибуте класса 
    self.environment
    """

    def __init__(self, filepath: str, number_of_obvs: int):
        df = pd.read_excel(f'../backend/{filepath}', index_col=0)
        df = df[~df['description'].duplicated()]
        wrapper = textprocessing.preprocessingWrapper()
        df['preprocessed_desc'] = wrapper(df, 'description')
        targets = df['industry'].str.lower().str.replace("'",'').str.replace('[','').str.replace(']','') \
            .str.replace('\s*','',regex=True).str.split(',')[~df['industry'].isna()]
        targets = targets.explode().unique()

        self.names2idx = {0: 'геймдев', 1: 'акварель'}
        self.actions2idx = {i: action for i, action in enumerate(targets)}

        obvs_r = np.random.uniform(-1,1,number_of_obvs)
        obvs_s = np.array([np.random.randint(18,39,number_of_obvs), np.random.randint(0,1,number_of_obvs)]).T
        obvs_a = np.random.randint(0,targets.shape[0]-1,number_of_obvs)
        
        self.environment = pd.DataFrame(zip(obvs_s,obvs_a,obvs_r),columns=['state','action','reward'])