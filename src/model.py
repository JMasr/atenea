import os
import uuid
import shutil
import pickle
import warnings
import itertools
import subprocess
import multiprocessing

import torch.nn as nn
import torch.nn.functional as function
import torch.utils.data

from tqdm import tqdm
from pyannote.core import Timeline, Segment, Annotation

from config import *
from data import Multimedia, Transcription, Dataset, Subtitle


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

    dir_path = "/".join(result_path.split("/")[:-1])
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))

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


class ALR(Model):
    def __init__(self, model_id):
        super().__init__(model_id, 'ALR')
        self.model = ALR_MODELS[self.model_id]

    def identify_language(self):
        return 'gl'


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

    def apply(self, multimedia=None):
        """
        This method is used for splitting long audios into speech chunks using a VAD or SAD system

        Parameters
        ----------
        multimedia: list with two objects
              List with the audio as a tensor in index-0 and sample rate in index-1

        Returns
        ----------
        speech_segments:    list of dicts
                            List containing ends and beginnings of speech chunks in milliseconds.
        acoustic_segments:  list of dicts
                            List containing ends and beginnings of non speech chunks in milliseconds.
        """
        wav_tensor = multimedia.get_wav_tensor()
        sampling_rate = multimedia.get_sampling_rate()

        if self.model_id == 'base':
            speech_segments = self.run_model_base(wav_tensor, sampling_rate)
            speech_segments = transform_frame_to_milliseconds(speech_segments, sampling_rate)
        elif self.model_id == 'gtm':
            path_out = multimedia.get_path().replace('.wav', '_output')
            subprocess.call([self.model, multimedia.get_path(), path_out, './pykaldi/sad'])
            results_path = f'{path_out}/{multimedia.get_name().replace(".wav", "")}/{multimedia.get_name().replace(".wav", ".sad")}'
            speech_segments = post_processing_scripting_models_results(results_path)
        else:
            raise TypeError("VAD model id isn't valid.")

        length_wav = len(wav_tensor)
        acoustic_segments = make_acoustic_segments_from_speech_segments(speech_segments, length_wav)

        return data_to_segment(speech_segments), data_to_segment(acoustic_segments)


class Diarization(Model):
    def __init__(self, model_id, thresholds=None):
        super().__init__(model_id, 'Diarization System')
        self.model = SPK_DIARIZATION_MODELS[self.model_id]
        self.drz_thresholds = None if not thresholds else thresholds.split(';')

        self.spk = {}
        self.diarization_annotations = Annotation()

    def get_spk(self):
        return self.spk

    def get_data(self):
        return self.data

    def set_spk(self, spk):
        self.spk = spk

    def set_data(self, data):
        self.data = data

    def run_model_gtm(self, multimedia):
        if self.drz_thresholds:
            th1_1st, th2_1st = self.drz_thresholds[0], self.drz_thresholds[1]
            window, period = self.drz_thresholds[2], self.drz_thresholds[3]
        else:
            th1_1st, th2_1st, window, period = '13.5', '14.5', '5.', '.5'

        path_out = multimedia.get_path().replace('.wav', '_drz_output')
        subprocess.call([self.model, multimedia.get_path(), path_out, th1_1st, th2_1st, window, period])

        path_results = f"{path_out}/{multimedia.get_name().replace('.wav', '_drz.rttm')}"
        with open(path_results, 'r', encoding="ISO-8859-1") as f:
            spk_annotations = f.readlines()

        for annotation in spk_annotations:
            annotation = annotation.split()
            time_init = float(annotation[3])
            time_ends = time_init + float(annotation[4])
            spk_id = annotation[7]

            s = Segment(time_init, time_ends)
            self.diarization_annotations[s] = spk_id

    def run_model_base(self, multimedia):
        spk_annotation = self.model({'audio': multimedia.get_path()})
        self.spk = spk_annotation.labels()

        return spk_annotation

    def apply(self, multimedia):
        if self.model_id == 'base':
            self.diarization_annotations = self.run_model_base(multimedia)
        elif self.model_id == 'gtm':
            self.run_model_gtm(multimedia)
        else:
            raise TypeError("Diarization model-id isn't valid.")

        return self.diarization_annotations


class ASR(Model):
    def __init__(self, model_id):
        super().__init__(model_id, 'ASR System')
        self.model = ASR_MODELS[self.model_id]

    def run_gtm(self, multimedia):
        path_out = multimedia.get_path().replace('.wav', '_output')
        subprocess.call([self.model, multimedia.get_path(), path_out, './pykaldi/asr'])
        multimedia_ctm = f"{path_out}/{multimedia.get_name().replace('.wav', '.ctm')}"
        return Transcription(multimedia_ctm)

    def apply(self, multimedia):

        if 'gtm' in self.model_id:
            transcription = self.run_gtm(multimedia)
        else:
            raise TypeError("ASR model-id isn't valid.")

        return transcription


class DeepPunctuation(nn.Module):
    def __init__(self, model_id, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuation, self).__init__()

        self.model_id = model_id
        self.model_path = ACPRS_MODELS[self.model_id][0]

        self.seq_length = ACPRS_MODELS[self.model_id][5]
        self.batch_sz = ACPRS_MODELS[self.model_id][6]
        self.output_dim = len(punctuation_dict)

        self.bert_layer = ACPRS_MODELS[self.model_id][1].from_pretrained(self.model_path)
        bert_dim = ACPRS_MODELS[self.model_id][3]

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim

        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=len(punctuation_dict))

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x


class ACPRS(Model):
    def __init__(self, model_id):
        super().__init__(model_id, 'Automatic Capitalization & Punctuation Restoration System')
        self.model_path = ACPRS_MODELS[self.model_id][0]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepPunctuation(self.model_id)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_path + 'weights.pt'))

        self.seq_length = ACPRS_MODELS[self.model_id][5]
        self.batch_sz = ACPRS_MODELS[self.model_id][6]

        with open('../models/dict.pkl', 'rb') as f:
            self.dict_cap_words = pickle.load(f)

    def get_final_transcription_and_confidence(self, y_conf: list, y_str: list):
        """
        Pre-process the sub-words and tokens output of the model to obedient the final list of word and confidence.

        Parameters
        ----------
        y_str: list
            List with all the sub-words and tokens predicted by the model.
        y_conf: list
            List with the prediction confidence of the model for each element in y_str.

        Returns
        ----------
        new_text: List with all the words of the transcription with capitalization and punctuation marks.
        new_confidence: List with the prediction confidence of the model for each word.
        """

        ind, new_text, new_confidence, join_word = 0, [], [], False
        while ind < len(y_str) - 1:
            if y_str[ind] in ['£', '¢', '[pad]', '[PAD]']:
                ind += 1
                continue
            elif (ind != 0) and ("#" in y_str[ind]):
                new_text[-1] = new_text[-1] + y_str[ind][2:]
                new_confidence[-1] = max(y_conf[ind], y_conf[ind - 1])
                ind += 1
                continue
            elif (ind != len(y_str) - 1) and ("#" in y_str[ind + 1]):
                new_t = y_str[ind] + y_str[ind + 1][2:]
                new_c = max(y_conf[ind], y_conf[ind + 1])
                ind += 2
            else:
                new_t = y_str[ind]
                new_c = y_conf[ind]
                ind += 1

            if not new_text or new_t.upper() in self.dict_cap_words or new_t[:-1].upper() in self.dict_cap_words:
                new_t = new_t[0].upper() + new_t[1:]
                new_c = 1.

            if join_word:
                join_word = False
                new_text[-1] += new_t

            elif new_t == '-':
                join_word = True
                new_text[-1] += new_t

            else:
                new_text.append(new_t) if new_t != '¡' else new_text.append('<UNK>')
                new_confidence.append(new_c)

        return new_text, new_confidence

    def inference(self, data_loader: torch.utils.data.DataLoader, tokenizer_inference):
        """
        Pass to the model the row transcription and get the capitalized and punctuated transcription inferred.

        Parameters
        ----------
        data_loader: DataLoader from torch
            Pytorch Objet with the raw transcription without capitalization or punctuation.

        tokenizer_inference: Tokenizer
            Transformer Object that transform words into tokens embeddings and vice versa.

        Returns
        ----------
        y_str: List with all the sub-words and tokens predicted by the model.
        y_conf: List with the prediction confidence of the model for each element in y_str.
       """
        y_str = []
        y_conf = []
        num_iteration = 0

        deep_punctuation = self.model.eval()
        with torch.no_grad():
            for x, y, att, y_mask in tqdm(data_loader, desc='test'):

                x, y, att, y_mask = x.to(self.device), y.to(self.device), att.to(self.device), y_mask.to(self.device)
                y_predict = deep_punctuation(x, att)
                logits = torch.nn.functional.softmax(y_predict, dim=1)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)

                batch_conf = []
                for b in range(logits.size()[0]):
                    for s in range(logits.size()[1]):
                        batch_conf.append(torch.max(logits[b, s, :]).item())

                num_iteration += 1
                x_tokens = tokenizer_inference.convert_ids_to_tokens(x.view(-1))
                for index in range(y.shape[0]):
                    x_tokens[index] = transformation_dict[y_predict[index].item()](x_tokens[index])
                y_str.append(x_tokens)
                y_conf.append(batch_conf)

        y_str = list(itertools.chain.from_iterable(y_str))
        y_conf = list(itertools.chain.from_iterable(y_conf))

        return y_str, y_conf

    def apply(self, input_text: list):
        """
        Apply the ACRS to an input raw transcription.

        Parameters
        ----------
        input_text: list
            List of each word of the raw transcription.

        Returns
        ----------
        text: List with all the sub-words and tokens predicted by the model.
        confidence: List with the prediction confidence of the model for each element in y_str.
        """

        tokenizer_inference = ACPRS_MODELS[self.model_id][2].from_pretrained(self.model_path)
        token_style = ACPRS_MODELS[self.model_id][4]

        test_set = [Dataset(input_text, tokenizer_c=tokenizer_inference, sequence_len=self.seq_length,
                            token_style=token_style, is_train=False)]

        # Data Loaders
        data_loader_params = {'batch_size': self.batch_sz, 'shuffle': False, 'num_workers': 0}
        test_loaders = [torch.utils.data.DataLoader(utt, **data_loader_params) for utt in test_set]

        y_str, y_conf = self.inference(test_loaders[0], tokenizer_inference)
        text, confidence = self.get_final_transcription_and_confidence(y_conf, y_str)

        return text, confidence


class Pipeline:
    def __init__(self, arguments):
        self.arguments = arguments
        self.name = arguments.name
        self.id = uuid.uuid4()

        self.alr = ALR(self.arguments.alr_model)
        self.vad = VAD(self.arguments.vad_model)
        self.drz = Diarization(self.arguments.diarization_model, self.arguments.drz_thresholds)

        self.language = self.alr.identify_language() if self.arguments.language == 'UNK' else arguments.language
        self.asr = ASR(f"{self.arguments.asr_model}-{self.language}")
        self.acpr = ACPRS(f"{self.arguments.asr_model}-{self.language}")

        self.multimedia = Multimedia(self.arguments.data_path, self.language)
        self.trans: Transcription = None

    def apply_vad(self):
        speech_segments, acoustic_segments = self.vad.apply(self.multimedia)
        self.multimedia.set_speech(speech_segments)
        self.multimedia.set_acoustic(acoustic_segments)
        return speech_segments, acoustic_segments

    def apply_diarization(self):
        print('Diarization!')
        spk_drz = self.drz.apply(self.multimedia)
        self.multimedia.annotations = spk_drz
        return spk_drz

    def apply_asr(self):
        print('ASR')
        asr_trans = self.asr.apply(self.multimedia)
        self.trans = asr_trans
        return asr_trans

    def apply_acpr(self):
        new_transcription, acpr_confidences = self.acpr.apply(self.trans.get_text().split())
        self.trans.update_transcription(words=new_transcription, acpr_confs=acpr_confidences)

    def make_srt(self):
        self.trans.update_spk_from_annotations(self.multimedia.annotations)
        subtitle = Subtitle(self.trans.get_annotation())
        subtitle.write_srt(self.multimedia.path)

    def run_models(self):

        def delete_dir(dir_path):
            if os.path.exists(dir_path):
                os.remove(dir_path)
            else:
                print("The file does not exist")

        def read_drz_results(multimedia):
            path_out = multimedia.get_path().replace('.wav', '_drz_output')
            path_results = f"{path_out}/{multimedia.get_name().replace('.wav', '_drz.rttm')}"
            with open(path_results, 'r', encoding="ISO-8859-1") as f:
                spk_annotations = f.readlines()

            diarization_annotations = Annotation()
            for annotation in spk_annotations:
                annotation = annotation.split()
                time_init = float(annotation[3])
                time_ends = time_init + float(annotation[4])
                spk_id = annotation[7]

                s = Segment(time_init, time_ends)
                diarization_annotations[s] = spk_id
            # delete_dir(path_out)

            return diarization_annotations

        def read_asr_results(multimedia):
            path_out = multimedia.get_path().replace('.wav', '_output')
            multimedia_ctm = f"{path_out}/{multimedia.get_name().replace('.wav', '.ctm')}"
            # delete_dir(path_out)

            return Transcription(multimedia_ctm)

        process_drz = multiprocessing.Process(name='drz', target=self.apply_diarization)
        process_asr = multiprocessing.Process(name='asr', target=self.apply_asr)
        process_drz.start()
        process_asr.start()

        while process_asr.is_alive():
            pass
        self.trans = read_asr_results(self.multimedia)
        self.apply_acpr()

        while process_drz.is_alive():
            pass
        self.multimedia.annotations = read_drz_results(self.multimedia)

