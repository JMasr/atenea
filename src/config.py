import os
import torch
import requests
from transformers import *


def download_file_from_google_drive(id_token, destination):
    """
    Download a file from Google Drive using its unique token `id` and write down in disk `destination`.
    :param id_token: Token of the Google Drive file.
    :type id_token: str
    :param destination: path were the Google Drive file will be saved.
    :type destination: str
    :return: -
    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, path_to_save):
        chunk_size = 32768
        os.makedirs(path_to_save, exist_ok=True)
        with open(path_to_save, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    resp = session.get(url, params={'id': id_token}, stream=True)
    token = get_confirm_token(resp)

    if token:
        params = {'id': id_token, 'confirm': token}
        resp = session.get(url, params=params, stream=True)

    save_response_content(resp, destination)


def check_for_component(destination, file_drive_id):
    if os.path.exists(destination) is False:
        print("Download component...")
        download_file_from_google_drive(file_drive_id, destination)


TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}

# 'O' -> No punctuation
punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3, 'ALL_CAPITAL': 4, 'FRITS_CAPITAL': 5,
                    'ALL_CAPITAL+COMMA': 6, 'ALL_CAPITAL+PERIOD': 7, 'ALL_CAPITAL+QUESTION': 8,
                    'FRITS_CAPITAL+COMMA': 9, 'FRITS_CAPITAL+PERIOD': 10, 'FRITS_CAPITAL+QUESTION': 11}

transformation_dict = {0: lambda x: x.lower(), 1: (lambda x: x + ','), 2: (lambda x: x + '.'),
                       3: (lambda x: x + '?'),
                       4: lambda x: x.upper(), 5: (lambda x: x[0].upper() + x[1:]), 6: (lambda x: x.upper() + ','),
                       7: (lambda x: x.upper() + '.'), 8: (lambda x: x.upper() + '?'),
                       9: (lambda x: x[0].upper() + x[1:] + ','), 10: (lambda x: x[0].upper() + x[1:] + '.'),
                       11: (lambda x: x[0].upper() + x[1:] + '?')}


# pretrained model name: (model class, model tokenizer, output dimension, token style)
ALR_MODELS = {'base': None,
              'gtm': None}

VAD_MODELS = {'base': torch.hub.load(repo_or_dir="../models/basic_vad", source='local',
                                     model='silero_vad', force_reload=True),
              'gtm': './pykaldi/sad/SAD.sh'}

SPK_DIARIZATION_MODELS = {'base': torch.hub.load('pyannote/pyannote-audio', 'dia'),
                          'gtm': './pykaldi/drz/procesaFichero.sh'}

ASR_MODELS = {'base-gl': None,
              'gtm-gl': './pykaldi/asr/asr.sh'}

ACPRS_MODELS = {'base-gl': None,
                'gtm-gl': ('../models/bertinho/', BertModel, AutoTokenizer, 768, 'bert', 96, 8),
                'gtm-spn': ('../models/berto/', BertModel, AutoTokenizer, 768, 'bert', 256, 8)}
