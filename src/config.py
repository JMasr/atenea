import os
import torch
import requests


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


# pretrained model name: (model class, model tokenizer, output dimension, token style)
VAD_MODELS = {'base': torch.hub.load(repo_or_dir="../models/basic_vad",
                                     source='local', model='silero_vad', force_reload=True),
              'gtm-base': './pykaldi/sad/SAD.sh'}

SPK_DIARIZATION_MODELS = {'base': None}
