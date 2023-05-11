from agent.dqn_net import DQNSolver
import torch
import numpy as np
from utils.config_loader import load_config
from fastapi.middleware.cors import CORSMiddleware
from utils import textprocessing
from utils.os_ops import json_read
from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel
from fastapi import FastAPI
from backend.cursor import RECORDS, load_stemmed_descriptions

# config & meta data
CONFIG_PATH = 'config.json'
META_PATH = 'api_meta.json'
CONFIG = load_config(CONFIG_PATH)
META_DATA = json_read(META_PATH)

# DQN
ACTION_SPACE = len(CONFIG['a2idx'])
PRETRAINED_W_PATH = './agent/DQN.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET = load_stemmed_descriptions()

# рекомендации
PREPROCESSER = textprocessing.preprocessingWrapper()
VECTORIZER = TfidfVectorizer(analyzer='word', decode_error='ignore')

app = FastAPI(
    title=META_DATA['CLIENT_META']['title'],
    description=META_DATA['CLIENT_META']['description'],
    version=META_DATA['CLIENT_META']['version'],
    terms_of_service=None,
    contact=META_DATA['CLIENT_META']['contact'],
    license_info=None
)

# schema-like объект для создания записей о состоянии среды
class envAddition(BaseModel):
    cum_reward: int
    age: int
    skill: str
    action: int

# CORS для удобства отладки
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post('/api/write/')
def write(addition: envAddition):
    """
    POST-эндпоинт, позволяющий агенту запоминать события в среде для дальнейшего оффлайн-переобучения
    """
    if addition.cum_reward > 5 or addition.cum_reward < -5 or addition.cum_reward == 0:
        return {
            'status_code': 422,
            'details': 'неверное значение кумулятивной награды'
        }
    RECORDS.insert_one({
        'state': [addition.age, CONFIG['idx2skills'][addition.skill]],
        'action': addition.action,
        'reward': np.tanh(addition.cum_reward)
    })
    return {
        'status_code': 200,
    }

@app.get('/api/recommend/')
async def recommend(age: int, skill: str, preference: str):
    """
    GET-эндпоинт, при запросе на который DQN-агент делает рекомендацию
    """
    if age < 14 or age > 100 or skill not in CONFIG['idx2skills'].keys() or len(preference) > 256:
        return {
            'status_code': 422,
            'details': 'некорректные параметры'
        }

    # DQN-рекомендация
    dqn_net = DQNSolver(n_actions=ACTION_SPACE, obvs_size=None).to(DEVICE)
    dqn_net.load_state_dict(torch.load(PRETRAINED_W_PATH, map_location=torch.device(DEVICE)))
    agent_actions = dqn_net(torch.Tensor(
        [age, CONFIG['idx2skills'][skill]]
    ))
    agent_best_action = agent_actions.argmax(0).item()
    agent_recommendation = CONFIG['a2idx'][agent_best_action]

    # сортировка рекомендованых вакансий относительно предпочтений пользователя
    preprocessed_preference = PREPROCESSER(preference)
    important_vacancies = DATASET[DATASET[agent_recommendation] == 1]
    vectorized = VECTORIZER.fit_transform(important_vacancies['desc']).toarray()
    vectorized_pref = VECTORIZER.transform([preprocessed_preference]).toarray()
    distances = 1 - np.sum(vectorized_pref * vectorized, axis=1) / (
            np.linalg.norm(vectorized, axis=1) * np.linalg.norm(vectorized_pref))
    args_of_max = np.argsort(distances)

    return {
        'status_code': 200,
        'details': {
            'action': agent_best_action,
            'recommended': important_vacancies.iloc[args_of_max[:min(args_of_max.shape[0], 5)], [0,1,3,4]].to_dict(
                orient='records'
            )
        }
    }