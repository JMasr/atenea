import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for the automatic rich-transcription run')
    # General arguments about the experiment (id, type of media, gpu-cpu, seed, useful directories, pipe-line)
    parser.add_argument('--name', default='test-00', type=str, help='name of run')
    parser.add_argument('--multimodal', default=False, type=bool, help='the data is a multimedia file')
    parser.add_argument('--cuda', default=0, type=int, help='use a cuda specific device or pass -1 to use cpu instead')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--data-path', default='../data/gam_08421_00349882273.wav', type=str, help='path to train/dev/test datasets')
    parser.add_argument('--save-path', default='/log/', type=str, help='model and log save directory')
    parser.add_argument('--pipe-line', default='default', type=str, help='pipe-line selected for the run')

    # Language Recognition Phase arguments (language, models, multilingual)
    parser.add_argument('--language', default='unk', type=str, help='language, available options are: gl, es, en, unk')
    parser.add_argument('--language-model', default='default', type=str, help='select the language recognition model')
    parser.add_argument('--multilingual', default=False, type=bool, help='there is more than one language present')

    # Diarization Phase arguments (models, number of speakers, acoustic events of interest)
    parser.add_argument('--vad-model', default='base', type=str, help='select a VAD or SAD model')
    parser.add_argument('--diarization-model', default='default', type=str, help='select the diarization model')
    parser.add_argument('--number-spks', default=0, type=int, help='number of speakers, if it is unknown introduce 0')
    parser.add_argument('--acoustic-events', default='all', type=str, help='interesting acoustic events to detect')

    # Speaker Metadata Phase arguments (models, biometrics, meta-information)
    parser.add_argument('--spk-model', default='default', type=str, help='select the spk model')
    parser.add_argument('--biometrics', default=True, type=bool, help='get all the biometric information possible')
    parser.add_argument('--meta-info', default=True, type=bool, help='get meta-information about spk such as emotions')

    # Audio Metadata arguments (enhancement, ...)

    # Video Metadata arguments

    args = parser.parse_args()
    return args
