import torch
import torchaudio
import matplotlib.pyplot as plt

from pyannote.core import Timeline, Annotation, Segment
from pyannote.core import notebook


def make_annotation_from_segments(segments: Segment, label: str, annotation: Annotation=None):
    if not annotation:
        annotation = Annotation()
    for s in segments:
        annotation[s] = label
    return annotation


class Audio(object):
    """
    path:   str
            Path to the audio file

    wav_tensor: torch.Tensor, 1-D
                One dimensional float torch.Tensor, that storage the audio file.
    sampling_rate: int
                Sampling rate of the audio file.
    """
    def __init__(self, audio_path: str):
        self.path = audio_path
        self.name = audio_path.split("/")[-1]

        self.wav_tensor, self.sampling_rate = self.read_as_tensor()

        self.speech_segments = Timeline()
        self.acoustic_segments = Timeline()

        self.annotations = Annotation()

    def __len__(self):
        return round(len(self.wav_tensor)/self.sampling_rate, 4)

    def get_path(self):
        return self.path

    def get_wav_tensor(self):
        return self.wav_tensor

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_speech(self):
        return self.speech_segments

    def get_acoustic(self):
        return self.acoustic_segments

    def set_path(self, path: str):
        self.path = path

    def set_wav_tensor(self, wav_t: torch.Tensor):
        self.wav_tensor = wav_t

    def set_sampling_rate(self, sr: int):
        self.sampling_rate = sr

    def set_speech(self, segments):
        self.speech_segments = segments

    def set_acoustic(self, segments):
        self.acoustic_segments = segments

    def show_speech(self):
        notebook.width = 10
        notebook.crop = Segment(0, self.__len__())
        plt.rcParams['figure.figsize'] = (notebook.width, 5)
        notebook.plot_timeline(self.speech_segments, time=True)
        plt.show()

    def show_acoustic(self):
        notebook.width = 10
        notebook.crop = Segment(0, self.__len__())
        plt.rcParams['figure.figsize'] = (notebook.width, 2)
        notebook.plot_timeline(self.acoustic_segments, time=True)
        plt.show()

    def show_vad(self):
        notebook.width = 10
        plt.rcParams['figure.figsize'] = (notebook.width, 2)
        notebook.crop = Segment(0, self.__len__())

        # plot annotation
        annotation = make_annotation_from_segments(self.speech_segments, 'Speaker')
        annotation = make_annotation_from_segments(self.acoustic_segments, 'noise', annotation)
        notebook.plot_annotation(annotation, legend=True, time=True)
        plt.show()

        with open("./vad.rttm", 'w') as f:
            annotation.write_rttm(f)

    def read_as_tensor(self):
        wav, sr = torchaudio.load(self.path)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        return wav.squeeze(0), sr

    def save_audio(self, output_path: str):
        torchaudio.save(output_path, self.wav_tensor.unsqueeze(0), self.sampling_rate)

    def resampling_tensor(self, new_freq: int):
        transform = torchaudio.transforms.Resample(orig_freq=self.sampling_rate, new_freq=new_freq)
        wav = transform(self.wav_tensor)
        return wav, new_freq

    def collect_chunks(self, list_time_steps: list):
        """
        This method segments and return specific chunks (in samples) in a one dimensional tensor.

        Parameters
        ----------
        list_time_steps: list of dicts, [{'start'}: 0, {'end'}: 123]
                         list containing ends and beginnings of audio chunks (in samples)

        Returns
        ----------
        chunks: torch.Tensor, one dimensional
                1-D tensor with all the chunks in a specific segment of audio
        """
        chunks = []
        for i in list_time_steps:
            chunks.append(self.wav[i['start']: i['end']])
        chunks = torch.cat(chunks)
        return chunks

    def chunks_to_seconds(self, frames):
        """
        This method converts the frames into its corresponding audio time in seconds.

        Parameters
        ----------
        frames: list of dicts, [{'start'}: 0, {'end'}: 123]
                list containing ends and beginnings of audio chunks (in samples)

        Returns
        ----------
        chunks: torch.Tensor, one dimensional
               1-D tensor with all the chunks in a specific segment of audio
        """

        for audio_dict in frames:
            audio_dict['start'] = round(audio_dict['start'] / self.sampling_rate, 1)
            audio_dict['end'] = round(audio_dict['end'] / self.sampling_rate, 1)
        return frames


class Video(Audio):
    def __init__(self, video_path):
        self.path = video_path


class RichTranscription(object):
    def __init__(self, multimedia_path: str):
        self.multimedia_path = multimedia_path

        self.spk = {}
        self.acu = []
        self.txt = []
        self.stm = []
        self.srt = []
