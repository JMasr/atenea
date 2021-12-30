import os

import torch
import torchaudio

import subprocess

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
    def __init__(self, model_id, wav_path):
        self.model_id = model_id
        self.model = VAD_MODELS[model_id]

        self.wav_path = wav_path
        self.wav_tensor, self.sample_rate = read_audio(wav_path)

        self.speech_timestamps, self.acoustic_events_timestamps = self.get_speech_timestamps(self.model)

    def get_speech_timestamps(self, model, min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 100,
                              return_seconds: bool = False):

        """
        This method is used for splitting long audios into speech chunks using a VAD or SAD system

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

        if self.model_id == 'base':

            audio, sampling_rate = self.wav_tensor, self.sample_rate

            if not torch.is_tensor(audio):
                try:
                    audio = torch.Tensor(audio)
                except:
                    raise TypeError("Audio cannot be casted to tensor. Cast it manually")

            if len(audio.shape) > 1:
                for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
                    audio = audio.squeeze(0)
                if len(audio.shape) > 1:
                    raise ValueError("More than 1 dimension in audio. Are you trying to process audio with 2 channels?")

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
                    value = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                    speech['end'] = value

            acoustic_events = []
            for i, speech in enumerate(speeches):
                if i == 0 and speech['start'] == 0:
                    continue
                elif i == 0 and speech['start'] != 0:
                    acoustic_events.append({'start': 0, 'end': speech['start']})
                else:
                    acoustic_events.append({'start': speeches[i-1]['end'], 'end': speech['start']})
                    if i == len(speeches) - 1 and speech['end'] != audio_length_samples:
                        acoustic_events.append({'start': speech['end'], 'end': audio_length_samples})

            if return_seconds:
                for speech_dict in speeches:
                    speech_dict['start'] = round(speech_dict['start'] / sampling_rate, 1)
                    speech_dict['end'] = round(speech_dict['end'] / sampling_rate, 1)
            elif step > 1:
                for speech_dict in speeches:
                    speech_dict['start'] *= step
                    speech_dict['end'] *= step

            for ind, t in enumerate(speeches):
                speeches[ind]['data'] = self.collect_chunks([speeches[ind]], self.wav_tensor)
                acoustic_events[ind]['data'] = self.collect_chunks([acoustic_events[ind]], self.wav_tensor)

            return speeches, acoustic_events

        elif self.model_id == 'gtm-base':

            try:
                cmd = f'{self.model} {self.wav_path} ../data/output'
                subprocess.run(cmd, check=True)
            except:
                raise TypeError("SAD model can't process this input. Please, check the paths!")
        else:
            raise TypeError("VAD model id isn't valid.")

    @staticmethod
    def collect_chunks(tss, wav: torch.Tensor):
        chunks = []
        for i in tss:
            chunks.append(wav[i['start']: i['end']])
        return torch.cat(chunks)

    def apply(self):
        return self.speech_timestamps, self.acoustic_events_timestamps

    def save_speech_segments(self, output_path=''):

        if output_path == '':
            output_path = "/".join(self.wav_path.split('/')[:-1])

        name = self.wav_path.split('/')[-1]

        output_path += '/vad/'
        os.makedirs(output_path, exist_ok=True)

        print("\nWriting Segments...")
        for ind, w in enumerate(self.speech_timestamps):
            tinit = round(w['start'] * 1000 / self.sample_rate)
            tend = round(w['end'] * 1000 / self.sample_rate)
            segment_name = output_path + f'{name}_vad_{tinit}-{tend}.wav'
            save_audio(segment_name, w['data'], sampling_rate=self.sample_rate)
            w['file_name'] = segment_name
            print(f"\t{segment_name}")


class RichTranscription:
    def __init__(self, arguments):
        self.arguments = arguments
        self.name = arguments.name
        self.id = uuid.uuid4()

        self.stm = []
        self.srt = []

    def apply_vad(self):
        vad = VAD(self.arguments.vad_model, self.arguments.data_path)
        return vad.apply()

    def save_speech(self, output=''):
        vad = VAD(self.arguments.vad_model, self.arguments.data_path)
        speech, _ = vad.apply()
        vad.save_speech_segments()


