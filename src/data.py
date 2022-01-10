import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt

from pympi.Elan import Eaf as Eaf_
from pyannote.core import Timeline, Annotation, Segment
from pyannote.core import notebook


def make_annotation_from_segments(time_line: Timeline, label: str, annotation: Annotation = None):
    if not annotation:
        annotation = Annotation()
    for s in time_line:
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
        for chunk in list_time_steps:
            chunks.append(self.wav_tensor[chunk['start']: chunk['end']])
        chunks = torch.cat(chunks)
        return chunks

    def chunks_to_seconds(self, segments):
        """
        This method converts the frames into its corresponding audio time in seconds.

        Parameters
        ----------
        segments: list of dicts, [{'start'}: 0, {'end'}: 16000]
                list containing ends and beginnings of audio chunks (in samples)

        Returns
        ----------
        segments: list of dicts, [{'start'}: 0, {'end'}: 1.0]
                  list containing ends and beginnings of audio chunks (in seconds)
        """

        for audio_dict in segments:
            audio_dict['start'] = round(audio_dict['start'] / self.sampling_rate, 4)
            audio_dict['end'] = round(audio_dict['end'] / self.sampling_rate, 4)
        return segments

    def seconds_to_chunks(self, segments):
        """
        This method converts the frames into its corresponding audio time in seconds.

        Parameters
        ----------
        segments: list of dicts, [{'start'}: 0, {'end'}: 1.0]
                  list containing ends and beginnings of audio chunks (in seconds)

        Returns
        ----------
        segments: list of dicts, [{'start'}: 0, {'end'}: 16000]
                list containing ends and beginnings of audio chunks (in frames)
        """
        for audio_dict in segments:
            audio_dict['start'] = round(audio_dict['start'] * self.sampling_rate, 4)
            audio_dict['end'] = round(audio_dict['end'] * self.sampling_rate, 4)
        return segments

    def make_vad_annotation(self, output_path: str = None):
        annotation = make_annotation_from_segments(self.speech_segments, 'Speaker')
        annotation = make_annotation_from_segments(self.acoustic_segments, 'noise', annotation)

        if not output_path:
            output_path = f"{self.path[:-4]}_vad.rttm"

        with open(output_path, 'w') as f:
            annotation.write_rttm(f)

        return annotation

    def show_vad(self):
        notebook.width = 10
        plt.rcParams['figure.figsize'] = (notebook.width, 2)
        notebook.crop = Segment(0, self.__len__())

        # plot annotation
        annotation = self.make_vad_annotation()
        notebook.plot_annotation(annotation, legend=True, time=True)
        plt.show()


class Video(Audio):
    def __init__(self, video_path: str):
        super().__init__(video_path)
        self.path = video_path


class RichTranscription(object):
    def __init__(self, multimedia_path: str):
        self.multimedia_path = multimedia_path

        self.spk = None
        self.acu = None
        self.txt = None
        self.stm = None
        self.srt = None
        self.eaf = None


class Eaf(object):
    def __init__(self, path):
        self.path = path
        self.eaf = Eaf_(path)

    @staticmethod
    def make_data_frame_from_tier(data_tier: list, df_in: pd.DataFrame = pd.DataFrame(), column: int = 2,
                                  fill_none_with: str = None):
        df_out = df_in.copy()
        for i in data_tier:
            time_start = int(i[0])
            time_final = int(i[1])
            value = i[2]

            if df_in.empty:
                entry = pd.Series([time_start, time_final, None, None, None, None, None, None])
                entry[column] = value
                df_out = df_out.append(entry, ignore_index=True)
            else:
                index = df_out.loc[(df_out['Time_Init'] == time_start) | (df_out['Time_End'] == time_final)].index
                if index.empty:
                    mach_tolerance = 350
                    mach = df_out.loc[(round(df_out['Time_Init']/mach_tolerance) == round(time_start/mach_tolerance)) |
                                      (round(df_out['Time_End']/mach_tolerance) == round(time_final/mach_tolerance))]
                df_out.iloc[mach.index, column] = value

        if fill_none_with:
            df_out.iloc[:, column] = df_out.iloc[:, column].fillna(fill_none_with)

        df_out.columns = ["Time_Init", "Time_End", "Text", "Text_Orto", "Speaker", "Language", "Topic", "Acoustic_Info"]
        return df_out

    def to_csv(self, path_csv: str = None):

        data = {}
        tiers_name = self.eaf.get_tier_names()
        for t in tiers_name:
            data[t] = self.eaf.get_annotation_data_for_tier(t)

        df_speech = self.make_data_frame_from_tier(data['Segmento'])
        df_speech = self.make_data_frame_from_tier(data["Speakers"], df_speech, column=4, fill_none_with="UNK")
        df_speech = self.make_data_frame_from_tier(data["Lingua"], df_speech, column=5, fill_none_with="galego")
        df_speech = self.make_data_frame_from_tier(data["Topics"], df_speech, column=6)

        df_acoustic_events = self.make_data_frame_from_tier(data["outros"], column=7)

        if not path_csv:
            path_csv = self.path.replace(".eaf", ".csv")
        df_speech.to_csv(path_csv)
        df_acoustic_events.to_csv(path_csv.replace(".csv", "_acoustic.csv"))

        return df_speech, df_acoustic_events