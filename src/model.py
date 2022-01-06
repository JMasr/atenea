import uuid
import warnings
import subprocess
from pyannote.core import Timeline, Segment

from config import *

from data import Audio
from data import RichTranscription


def transform_frame_to_milliseconds(frames, sampling_rate):
    for f in frames:
        f['start'] = int(f['start'] / sampling_rate * 1000)
        f['end'] = int(f['end'] / sampling_rate * 1000)
    return frames


def collect_chunks(tss: list, wav: torch.Tensor):
    """
    This method segments and return specific chunks (in samples) in a one dimensional tensor.

    Parameters
    ----------
    tss: list of dicts, [{'start'}: 0, {'end'}: 123]
                     list containing ends and beginnings of audio chunks (in samples)

    wav: torch.Tensor, one dimensional
         1-D float torch.Tensor, that storage the audio file.

    Returns
    ----------
    chunks: torch.Tensor, one dimensional
            1-D tensor with all the chunks in a specific segment of audio
    """
    chunks = []
    for i in tss:
        chunks.append(wav[i['start']: i['end']])
    return torch.cat(chunks)


def adding_frames_using_time_steps(segments, wav_tensor):
    for ind, t in enumerate(segments):
        segments[ind]['data'] = collect_chunks([segments[ind]], wav_tensor)
    return segments


def make_acoustic_segments_from_speech_segments(speech_segments, audio_length_samples):
    acoustic_events = []
    for i, speech in enumerate(speech_segments):
        if i == 0 and speech['start'] == 0:
            continue
        elif i == 0 and speech['start'] != 0:
            acoustic_events.append({'start': 0, 'end': speech['start']})
        else:
            acoustic_events.append({'start': speech_segments[i - 1]['end'], 'end': speech['start']})
            if i == len(speech_segments) - 1 and speech['end'] != audio_length_samples:
                acoustic_events.append({'start': speech['end'], 'end': audio_length_samples})

    if acoustic_events[-1]['end'] != audio_length_samples:
        acoustic_events.append({'start': speech_segments[-1]['end'], 'end': audio_length_samples})

    return acoustic_events


def check_time_steps_continuity(segments):
    output_segments = []
    for ind, s in enumerate(segments):
        act_start = int(s['start'])
        act_end = int(s['end'])

        if ind == 0:
            output_segments.append({'start': act_start, 'end': act_end})
        elif 0 < ind < len(segments):
            if act_start == output_segments[-1]['end']:
                output_segments[-1]['end'] = act_end
            else:
                output_segments.append({'start': act_start, 'end': act_end})

    return output_segments


def post_processing_scripting_models_results(result_path):
    with open(result_path, 'r') as f:
        result = f.readlines()

    speech_segments = [{'start': segment.split()[0], 'end': segment.split()[1]} for segment in result if'sp' in segment]
    speech_segments = check_time_steps_continuity(speech_segments)
    return speech_segments


def data_to_segment(data):
    time_line = Timeline()
    Segment.set_precision(4)
    for d in data:
        time_line.add(Segment(d['start']/1000, d['end']/1000))
    return time_line


def checks_and_warnings2(sampling_rate, wav_tensor, window_size_samples):
    if not torch.is_tensor(wav_tensor):
        raise TypeError("Audio cannot be casted to tensor.")

    if len(wav_tensor.shape) > 1:
        for i in range(len(wav_tensor.shape)):  # trying to squeeze empty dimensions
            wav_tensor = wav_tensor.squeeze(0)
        if len(wav_tensor.shape) > 1:
            raise ValueError("More than 1 dimension in audio. Are you trying to process audio with 2 channels?")

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        wav_tensor = wav_tensor[::step]
        warnings.warn('Sampling rate is a multiply of 16000, casting to 16000 manually!')
    else:
        step = 1

    if sampling_rate == 8000 and window_size_samples > 768:
        warnings.warn(
            'window_size_samples is too big for 8000 sampling_rate! '
            'Better set window_size_samples to 256, 512 or 768 for 8000 sample rate!')

    if window_size_samples not in [256, 512, 768, 1024, 1536]:
        warnings.warn(
            'Unusual window_size_samples! Supported window_size_samples:\n'
            ' - [512, 1024, 1536] for 16000 sampling_rate\n - [256, 512, 768] for 8000 sampling_rate')

    return sampling_rate, step, wav_tensor


def calculate_speech_timesteps_from_probs(speech_probs, audio_length_samples, min_silence_samples, min_speech_samples,
                                          threshold, window_size_samples):
    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0
    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = window_size_samples * i
            continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                temp_end = 0
                current_speech = {}
                triggered = False
                continue
    if current_speech:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)
    return speeches


def post_processing_timesteps(speeches, audio_length_samples, speech_pad_samples, step):
    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))

        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - silence_duration // 2))
            else:
                speech['end'] += int(speech_pad_samples)
        else:
            value = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
            speech['end'] = value
    if step > 1:
        for speech_dict in speeches:
            speech_dict['start'] *= step
            speech_dict['end'] *= step

    return speeches


class Model(object):
    def __init__(self, model_id, model_type):
        self.model = None
        self.model_id = model_id
        self.model_type = model_type

    def __str__(self):
        return f'Model ID: {self.model_id}\nModel Type: {self.model_type}'

    def get_model(self):
        return self.model

    def get_model_id(self):
        return self.model_id

    def get_model_type(self):
        return self.model_type

    def set_model(self, model):
        self.model = model

    def set_model_id(self, model_id):
        self.model_id = model_id

    def set_model_type(self, model_type):
        self.model_type = model_type

    def load_model(self):
        pass

    def train(self, datasets):
        pass

    def apply(self, data):
        pass


class VAD(Model):
    def __init__(self, model_id):
        super().__init__(model_id, 'VAD')
        self.model = VAD_MODELS[self.model_id]

    def calculate_speech_frame_probs(self, audio_length_samples, sampling_rate, wav_tensor, window_size_samples):
        speech_probs = []
        for current_start_sample in range(0, audio_length_samples, window_size_samples):
            chunk = wav_tensor[current_start_sample: current_start_sample + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
            speech_prob = self.model(chunk, sampling_rate).item()
            speech_probs.append(speech_prob)
        return speech_probs

    def run_model_base(self, wav_tensor, sampling_rate):

        threshold = 0.75  # Speech threshold: float (default - 0.5)
        speech_pad_ms = 30  # int (default - 30 ms) Final speech chunks are padded by this value each side.
        window_size_samples = 1536  # int (default - 1536 samples) Audio chunks size.

        min_speech_duration_ms = 250
        min_silence_duration_ms = 100

        min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
        min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        speech_pad_samples = sampling_rate * speech_pad_ms / 1000

        audio_length_samples = len(wav_tensor)

        sampling_rate, step, wav_tensor = checks_and_warnings2(sampling_rate, wav_tensor, window_size_samples)

        self.model.reset_states()
        speech_probs = self.calculate_speech_frame_probs(audio_length_samples, sampling_rate,
                                                         wav_tensor, window_size_samples)

        speeches = calculate_speech_timesteps_from_probs(speech_probs, audio_length_samples, min_silence_samples,
                                                         min_speech_samples, threshold, window_size_samples)

        speeches = post_processing_timesteps(speeches, audio_length_samples, speech_pad_samples, step)

        return speeches

    def apply(self, data=None):
        """
        This method is used for splitting long audios into speech chunks using a VAD or SAD system

        Parameters
        ----------
        data: list with two objects
              List with the audio as a tensor in index-0 and sample rate in index-1

        Returns
        ----------
        speech_segments:    list of dicts
                            List containing ends and beginnings of speech chunks in milliseconds.
        acoustic_segments:  list of dicts
                            List containing ends and beginnings of non speech chunks in milliseconds.
        """
        wav_tensor = data.get_wav_tensor()
        sampling_rate = data.get_sampling_rate()

        if self.model_id == 'base':
            speech_segments = self.run_model_base(wav_tensor, sampling_rate)
            speech_segments = transform_frame_to_milliseconds(speech_segments, sampling_rate)
        elif self.model_id == 'gtm-base':
            subprocess.call([self.model, data.get_path(), '../data/output', './pykaldi/sad'])
            results_path = f'../data/output/{data.name[:-4]}/{data.name[:-4]}.sad'
            speech_segments = post_processing_scripting_models_results(results_path)
        else:
            raise TypeError("VAD model id isn't valid.")

        length_wav = len(wav_tensor)
        acoustic_segments = make_acoustic_segments_from_speech_segments(speech_segments, length_wav)

        return data_to_segment(speech_segments), data_to_segment(acoustic_segments)


class Diarization(Model):
    def __init__(self, model_id):
        super().__init__(model_id, 'Diarization System')
        self.model = SPK_DIARIZATION_MODELS[self.model_id]

        self.spk = {}
        self.data = None

    def get_spk(self):
        return self.spk

    def get_data(self):
        return self.data

    def set_spk(self, spk):
        self.spk = spk

    def set_data(self, data):
        self.data = data

    def apply(self, new_data):
        pass


class Pipeline:
    def __init__(self, arguments):
        self.arguments = arguments
        self.name = arguments.name
        self.id = uuid.uuid4()

        self.audio = Audio(self.arguments.data_path)
        self.trans = RichTranscription(self.arguments.data_path)
        self.vad = VAD(self.arguments.vad_model)
        self.diaz = Diarization(self.arguments.diarization_model)

    def apply_vad(self):
        speech_segments, acoustic_segments = self.vad.apply(self.audio)
        self.audio.set_speech(speech_segments)
        self.audio.set_acoustic(acoustic_segments)
        return speech_segments, acoustic_segments

    def apply_diarization(self):
        pass
