import os

import torch
import torchaudio

import re
import uuid
import warnings

from config import VAD_MODELS


def read_audio(path: str, sampling_rate: int = 16000):

        wav, sr = torchaudio.load(path)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                       new_freq=sampling_rate)
            wav = transform(wav)
            sr = sampling_rate

        assert sr == sampling_rate
        return wav.squeeze(0), sr


def save_audio(path: str, tensor: torch.Tensor, sampling_rate: int = 16000):
    torchaudio.save(path, tensor.unsqueeze(0), sampling_rate)


def read_video(path: str):
    pass


class VAD:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = VAD_MODELS[model_id][0]

    def get_speech_timestamps(self, audio: torch.Tensor, model, sampling_rate: int = 16000,
                              min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 100,
                              return_seconds: bool = False):

        """
        This method is used for splitting long audios into speech chunks using silero VAD

        Parameters
        ----------
        audio: torch.Tensor, one dimensional
            One dimensional float torch.Tensor, other types are casted to torch if possible

        model: preloaded .jit silero VAD model

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_speech_duration_ms: int (default - 250 milliseconds)
            Final speech chunks shorter min_speech_duration_ms are thrown out

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)

        Returns
        ----------
        speeches: list of dicts
            list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
        """

        if not torch.is_tensor(audio):
            try:
                audio = torch.Tensor(audio)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        if len(audio.shape) > 1:
            for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
                audio = audio.squeeze(0)
            if len(audio.shape) > 1:
                raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

        if self.model_id == 'base':
            threshold = 0.75  # Speech threshold: float (default - 0.5)
            window_size_samples = 1536  # int (default - 1536 samples) Audio chunks size.
            speech_pad_ms = 30  # int (default - 30 ms) Final speech chunks are padded by this value each side.

            if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
                step = sampling_rate // 16000
                sampling_rate = 16000
                audio = audio[::step]
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

            model.reset_states()
            min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
            min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
            speech_pad_samples = sampling_rate * speech_pad_ms / 1000

            audio_length_samples = len(audio)

            speech_probs = []
            for current_start_sample in range(0, audio_length_samples, window_size_samples):
                chunk = audio[current_start_sample: current_start_sample + window_size_samples]
                if len(chunk) < window_size_samples:
                    chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
                speech_prob = model(chunk, sampling_rate).item()
                speech_probs.append(speech_prob)

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
                    speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

            if return_seconds:
                for speech_dict in speeches:
                    speech_dict['start'] = round(speech_dict['start'] / sampling_rate, 1)
                    speech_dict['end'] = round(speech_dict['end'] / sampling_rate, 1)
            elif step > 1:
                for speech_dict in speeches:
                    speech_dict['start'] *= step
                    speech_dict['end'] *= step

            return speeches
        else:
            raise TypeError("VAD model id isn't valid.")

    @staticmethod
    def collect_chunks(tss, wav: torch.Tensor):
        chunks = []
        for i in tss:
            chunks.append(wav[i['start']: i['end']])
        return torch.cat(chunks)

    def apply_2_wav(self, data_path: str, one_audio: bool = False):

        files = [data_path + f for f in os.listdir(data_path) if '.wav' in f]

        for f in files:

            sampling_rate = 16000  # also accepts 8000
            wav = read_audio(f, sampling_rate=sampling_rate)
            speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=sampling_rate)

            output_path = data_path + 'vad/'
            os.makedirs(output_path, exist_ok=True)
            if one_audio:
                # merge all speech chunks to one audio
                output_path = re.sub(r'\.wav', '_vad.wav', re.sub(data_path, output_path, f))
                save_audio(output_path, self.collect_chunks(speech_timestamps, wav), sampling_rate=sampling_rate)
            else:
                for ind, w in enumerate(speech_timestamps):
                    tinit = round(w['start'] * 1000 / sampling_rate)
                    tend = round(w['end'] * 1000 / sampling_rate)
                    temp_f = re.sub(r'\.wav', f'-vad_{tinit}-{tend}.wav', re.sub(data_path, output_path, f))
                    save_audio(temp_f, self.collect_chunks([speech_timestamps[ind]], wav), sampling_rate=sampling_rate)

    def apply_4_mem(self, wav: torch.Tensor, sampling_rate: int = 16000):
        speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=sampling_rate)
        for ind, t in enumerate(speech_timestamps):
            speech_timestamps[ind]['data'] = self.collect_chunks([speech_timestamps[ind]], wav)

        return speech_timestamps


class RichTranscription:
    def __init__(self, arguments):
        self.id = uuid.uuid4()
        self.origin_name = arguments.name
        self.language = arguments.language
        self.multimodal = arguments.multimodal

        self.wav, self.sampling_rate = read_audio(arguments.data_path)
        self.stm = []
        self.srt = []

        self.vad = VAD(arguments.vad_model)
        self.speech_timestamps = self.vad.apply_4_mem(self.wav, self.sampling_rate)