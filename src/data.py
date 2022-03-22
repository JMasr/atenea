import pickle

import time
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from moviepy.editor import VideoFileClip

from pympi.Elan import Eaf as Eaf_
from pyannote.core import Timeline, Annotation, Segment
from pyannote.core import notebook

from string import punctuation

non_end = ('a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia', 'hasta',
           'mediante', 'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras', 'versus', 'vía', 'y', 'e', 'ni', 'que',
           'o', 'u', 'ora', 'bien', 'tanto', 'como', 'cuanto', 'así', 'igual', 'mismo', 'sino', 'también', 'pero',
           'mas', 'empre', 'mientras', 'sino', 'o', 'u', 'ya', 'ora', 'fuera', 'sea', 'porque', 'como', 'dado', 'visto',
           'puesto', 'pues', 'si', 'sin', 'aunque', 'aún', 'mientras', 'salvo', 'luego', 'conque', 'ergo', 'solo',
           'siempre', 'nunca', 'el', 'la', 'lo', 'los', 'las', 'un', 'una', 'uno', 'unas', 'unos', 'esto', 'esta', 'estos',
           'estas', 'aquello', 'aquella', 'aquellos', 'aquellas', 'eso', 'esa', 'esos', 'esas', 'mi', 'su', 'tu', 'mis',
           'sus', 'tus', 'mío', 'míos', 'tuyo', 'tuyos', 'suyo', 'suyos', 'son',
           'unha', 'fóra', 'sen', 'do', 'dos', 'da', 'das', 'debaixo', 'despois', 'estou', 'embaixo', 'ao', 'ademais',
           'oposto', 'máis', 'dende', 'ata', 'a', 'sen', 'dúas', 'lonxe', 'petro', 'ademais', 'adiante', 'aqueles',)


def save_obj(obj, name=None):
    """Save a Python object into a `pickle` file on disk.

        Parameters
        ----------
        obj  : the Python object to be saved.
            Input Python object
        name : path where the Python object will be saved, without the .pkl extension.
            Input string
        Returns
        -------
        Nothing
    """
    if name is None:
        with open('obj/' + "obj_saved_" + str(time.time()) + '.pkl', 'wb') as f:
            pickle.dump(obj, f, 0)
    else:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, 0)


def load_obj(name):
    """Load a `pickle` object from disk.

        Parameters
        ----------
        name : path to the object without the .pkl extension.
            Input string
        Returns
        -------
        The Python object store in disk.
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Audio(object):
    """
    path:   str
            Path to the audio file

    wav_tensor: torch.Tensor, 1-D
                One dimensional float torch.Tensor, that storage the audio file.
    sampling_rate: int
                Sampling rate of the audio file.
    """
    def __init__(self, audio_path: str, language: str):
        self.path = audio_path
        self.name = audio_path.split("/")[-1]

        self.wav_tensor, self.sampling_rate = self.read_as_tensor()

        self.language: str = language
        self.speech_segments: Timeline = Timeline()
        self.acoustic_segments: Timeline() = Timeline

        self.annotations: Annotation = Annotation()

    def __len__(self):
        return round(len(self.wav_tensor)/self.sampling_rate, 4)

    def get_path(self):
        return self.path

    def get_name(self):
        return self.name

    def get_wav_tensor(self):
        return self.wav_tensor

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_language(self):
        return self.language

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

    @staticmethod
    def make_annotation_from_segments(time_line: Timeline, label: str, input_annotation: Annotation = None):
        if not input_annotation:
            input_annotation = Annotation()

        for s in time_line:
            input_annotation[s] = label
        return input_annotation

    def make_annotation(self, output_path: str = None):

        annotation = self.make_annotation_from_segments(self.speech_segments, 'SPEAKER')
        annotation = self.make_annotation_from_segments(self.acoustic_segments, 'noise', annotation)

        if not output_path:
            output_path = f"{self.path.replace('.wav', '_output')}/"
            os.makedirs(output_path, exist_ok=True)

        with open(f"{output_path}{self.name.replace('.wav', '_vad.rttm')}", 'w') as f:
            annotation.write_rttm(f)

        return annotation

    def update_annotations_with_speech(self, new_speech):
        new_speech = self.make_annotation_from_segments(self.acoustic_segments, 'noise', new_speech)
        self.annotations = new_speech

    def update_annotations_with_acoustic(self, new_acoustic):
        pass

    def show_annotation(self):
        notebook.width = 10
        plt.rcParams['figure.figsize'] = (notebook.width, 2)
        notebook.crop = Segment(0, self.__len__())

        # plot annotation
        if not self.annotations:
            self.annotations = self.make_annotation()

        notebook.plot_annotation(self.annotations, legend=True, time=True)
        plt.show()


class Multimedia(Audio):
    def __init__(self, media_path: str, language: str):
        _, self.ext = os.path.splitext(media_path)
        if self.ext != ".wav":
            self.v_path = media_path
            self.clip = VideoFileClip(self.v_path)
            self.extract_audio()

        super().__init__(media_path.replace(self.ext, ".wav"), language)

    def extract_audio(self):
        filename, _ = os.path.splitext(self.v_path)
        self.clip.audio.write_audiofile(f"{filename}.wav", fps=16000)


class Transcription(object):
    def __init__(self, multimedia_ctm):

        if isinstance(multimedia_ctm, list):
            self.ctm = multimedia_ctm
        elif isinstance(multimedia_ctm, str):
            with open(multimedia_ctm, 'r', encoding="ISO-8859-1") as f:
                self.ctm = f.readlines()

        self.times_init, self.times_ends, self.words, self.confidences = [], [], [], []
        for line in self.ctm:
            line = line.strip().split()
            self.times_init.append(float(line[2]))
            self.times_ends.append(float(line[3]))
            self.words.append(line[4])
            self.confidences.append(float(line[5]))

        self.text: str = " ".join(self.words)
        self.spk_ids: list = ['UNK'] * len(self.times_init)
        self.spk_confidences: list = [None] * len(self.times_init)
        self.acpr_confidences: list = self.spk_confidences.copy()

        self.annotation: dict = {}
        self.make_annotation()

    def get_ctm(self):
        return self.ctm

    def get_times_init(self):
        return self.times_init

    def get_times_end(self):
        return self.times_ends

    def get_words(self):
        return self.words

    def get_text(self):
        return self.text

    def get_annotation(self):
        return self.annotation

    def make_annotation(self):
        for ind in range(len(self.times_init)):
            self.annotation[self.times_init[ind]] = [self.times_ends[ind],
                                                     self.words[ind], self.confidences[ind], self.acpr_confidences[ind],
                                                     self.spk_ids[ind], self.spk_confidences[ind]]

    def update_transcription(self, times_init=None, times_ends=None, words=None, confidences=None, acpr_confs=None,
                             spk_ids=None, spk_confi=None):
        if times_init:
            self.times_init = times_init
        if times_ends:
            self.times_ends = times_ends
        if words:
            self.words = words
            self.text = " ".join(words)
        if confidences:
            self.confidences = confidences
        if acpr_confs:
            self.acpr_confidences = acpr_confs
        if spk_ids:
            self.spk_ids = spk_ids
        if spk_confi:
            self.spk_confidences = spk_confi

        self.make_annotation()

    def update_spk_from_annotations(self, annotation: Annotation):

        def extract_spk_ids(input_annotation):

            def make_segments_with_tolerance(time_init, time_ends, tolerance=0.1):
                s1 = Segment(time_init, time_ends)
                s2 = Segment(time_init + tolerance, time_ends)
                s3 = Segment(time_init, time_ends + tolerance)
                s4 = Segment(time_init + tolerance, time_ends + tolerance)
                s5 = Segment(time_init - tolerance, time_ends)
                s6 = Segment(time_init, time_ends - tolerance)
                s7 = Segment(time_init - tolerance, time_ends - tolerance)
                return [s1, s2, s3, s4, s5, s6, s7]

            def get_spk_id_from_segments(input_segments: list, annot: Annotation):
                for unk_segment in input_segments:
                    for segment, track, label in annot.itertracks(yield_label=True):
                        if unk_segment in segment:
                            return label
                return 'UNK'

            spk_ids = []
            for t_init, t_ends in zip(self.times_init, self.times_ends):
                segments = make_segments_with_tolerance(t_init, t_init + t_ends)
                spk_ids.append(get_spk_id_from_segments(segments, input_annotation))

            return spk_ids

        if len(annotation) == 1:
            self.spk_ids = [annotation.labels()[0]] * len(self.times_init)
        else:
            self.spk_ids = extract_spk_ids(annotation)
        self.make_annotation()


class Subtitle(object):
    def __init__(self, annotation: dict):
        self.annotation = annotation
        self.subtitle = self.make_subtitle()

    def make_lines_from_annotation(self):

        def split_1st_max_char_duration(annotation):
            lines, line = {1: ''}, []
            ind, line_duration, line_char_len = 0, 0, 0

            times = list(annotation.keys())
            start_time = times[0]
            start_spk = annotation[start_time][4]

            while ind < len(annotation):
                values = annotation[times[ind]]
                duration, word, spk = values[0], values[1], values[4]

                line_duration += duration
                line_char_len += len(word) + 1

                if ind == len(annotation) - 1 and spk == start_spk:
                    line.append(times[ind])
                    srt_indx = list(lines.keys())[-1]
                    lines[srt_indx] = line

                elif ind == len(annotation) - 1 and spk != start_spk:
                    srt_indx = list(lines.keys())[-1]
                    lines[srt_indx] = line
                    lines[srt_indx + 1] = [times[ind]]

                elif (line_duration <= 4 and line_char_len < 55) and spk == start_spk:
                    line.append(times[ind])

                    if word[-1] in punctuation:
                        start_time = times[ind + 1]
                        srt_indx = list(lines.keys())[-1]
                        lines[srt_indx] = line
                        lines[srt_indx + 1] = start_time

                        line = [start_time]
                        start_spk = annotation[start_time][4]
                        line_duration, line_char_len = 0, 0
                        ind += 1

                elif (line_duration >= 3 or line_char_len >= 55) or spk != start_spk:
                    start_time = times[ind]
                    if spk == start_spk and annotation[line[-1]][1] in non_end:
                        line.append(start_time)
                        start_time = times[ind + 1]
                        ind += 1

                    srt_indx = list(lines.keys())[-1]
                    lines[srt_indx] = line
                    lines[srt_indx + 1] = start_time

                    line = [start_time]
                    start_spk = annotation[start_time][4]
                    line_duration, line_char_len = 0, 0

                ind += 1

            return lines

        def split_2nd_build_lines(annotation, lines: dict):
            act_spk = annotation[lines[1][0]][4]
            ind, subtitles = 1, []
            while ind <= len(lines):
                words, asr_conf, punct_conf, spk_id = [], [], [], ''
                start_time_line = lines[ind][0]
                dur_last_word = annotation[lines[ind][-1]][0]
                end_time_line = lines[ind][-1] + dur_last_word

                for position, value in enumerate(lines[ind]):

                    asr_conf.append(annotation[value][2])
                    punct_conf.append(annotation[value][3])
                    spk_id = annotation[value][4]

                    word = annotation[value][1]
                    if position == 0 and spk_id != act_spk:
                        word = word[0].upper() + word[1:]
                        act_spk = spk_id
                    words.append(word)

                subtitles.append([start_time_line, end_time_line, words, asr_conf, punct_conf, spk_id])
                ind += 1

            return subtitles

        def split_3rd_refine(raw_subtitle: list):
            def sum_lines(line_1: list, line_2: list):
                line_1[1] = line_2[1]
                line_1[2].extend(line_2[2])
                line_1[3].extend(line_2[3])
                line_1[4].extend(line_2[4])

                return line_1

            ind = 0
            final_subtitle = []
            while ind < len(raw_subtitle):
                spk_id = raw_subtitle[ind][-1]
                if ind < len(raw_subtitle) - 1 and spk_id == raw_subtitle[ind + 1][-1] and len(
                        raw_subtitle[ind + 1][2]) <= 2:
                    new_line = sum_lines(raw_subtitle[ind], raw_subtitle[ind + 1])
                    final_subtitle.append(new_line)
                    ind += 2
                else:
                    final_subtitle.append(raw_subtitle[ind])
                    ind += 1

            return final_subtitle

        input_annotation = self.annotation.copy()
        raw_lines = split_1st_max_char_duration(input_annotation)
        raw_subtitles = split_2nd_build_lines(input_annotation, raw_lines)
        subtitle = split_3rd_refine(raw_subtitles)
        return subtitle

    @staticmethod
    def color_subtitle(uncolored_subtitle):
        def msec2srttime(msecs):
            secs, rmsecs = divmod(msecs, 1000)
            mins, secs = divmod(secs, 60)
            hours, mins = divmod(mins, 60)
            return '%02d:%02d:%02d,%03d' % (hours, mins, secs, rmsecs)

        def conf2color(srt_words, srt_conf, srt_conf_cap_punt):
            q = chr(34)
            yellow_font = '<font color=' + q + 'yellow' + q + '>'
            orange_font = '<font color=' + q + 'orange' + q + '>'
            red_font = '<font color=' + q + 'red' + q + '>'
            end_font = '</font>'

            def put_color(unc_word, x_conf):
                if x_conf > 0.9:
                    return unc_word
                elif x_conf > 0.7:
                    return yellow_font + unc_word + end_font
                elif x_conf > 0.5:
                    return orange_font + unc_word + end_font
                else:
                    return red_font + unc_word + end_font

            colered_line = ""
            for ind_word, word in enumerate(srt_words):
                if word.isalnum() and word == word.lower():  # just asr confident to lower case
                    colered_line += put_color(word, srt_conf[ind_word]) + " "
                elif word == word.lower() and not word.isalpha():
                    colered_line += put_color(word[:-1], srt_conf[ind_word])
                    colered_line += put_color(word[-1], srt_conf_cap_punt[ind_word]) + " "
                elif word[0] == word[0].upper() and word != word.upper():
                    colered_line += put_color(word[0], srt_conf_cap_punt[ind_word])
                    if word.isalnum():
                        colered_line += put_color(word[1:], srt_conf[ind_word]) + " "
                    else:
                        colered_line += put_color(word[1:-1], srt_conf[ind_word])
                        colered_line += put_color(word[-1], srt_conf_cap_punt[ind_word]) + " "
                else:
                    colered_line += put_color(word, min(srt_conf[ind_word], srt_conf_cap_punt[ind_word]))

            return colered_line

        colored_subtitle = []
        for ind, line in enumerate(uncolored_subtitle):
            colored_sub = conf2color(line[2], line[3], line[4])
            colored_sub = f'({line[-1]}) ' + colored_sub
            colored_subtitle.append([ind + 1, msec2srttime(line[0]*1000), msec2srttime(line[1]*1000), colored_sub])
        return colored_subtitle

    def make_subtitle(self):
        subtitle = self.make_lines_from_annotation()
        colored_lines = self.color_subtitle(subtitle)
        return colored_lines

    def write_srt(self, path_out):
        path_out = path_out.replace('.wav', '.srt')
        with open(path_out, 'w') as f:
            for line in self.subtitle:
                print(("\n%d\n%s --> %s\n%s\n" % (line[0], line[1], line[2], line[3])))
                f.write("\n%d\n%s --> %s\n%s\n" % (line[0], line[1], line[2], line[3]))


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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_txt, tokenizer_c, sequence_len, token_style, is_train=False):
        """

        :param input_txt: list containing tokens and punctuations separated by tab in lines
        :param tokenizer_c: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        :param augment_rate: token augmentation rate when preparing data
        :param is_train: if false do not apply augmentation
        """

        self.data = self.parse_data(input_txt, tokenizer_c, sequence_len, token_style)
        self.sequence_len = sequence_len
        self.token_style = token_style
        self.is_train = is_train

    @staticmethod
    def parse_data(lines, tokenizer_data, sequence_len, token_style):
        """

        :param lines: list that contains tokens and punctuations separated by tab in lines
        :param tokenizer_data: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: maximum length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        :return: list of [tokens_index, punctuation_index, attention_masks, punctuation_mask], each having sequence_len
        punctuation_mask is used to ignore special indices like padding and intermediate sub-word token during evaluation
        """
        data_items = []

        # loop until end of the entire text
        idx = 0
        while idx < len(lines):
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            y = [0]
            y_mask = [1]  # which positions we need to consider while evaluating i.e., ignore pad or sub tokens

            # loop until we have required sequence length
            # -1 because we will have a special end of sequence token at the end
            while len(x) < sequence_len - 1 and idx < len(lines):
                word, punc = lines[idx], 'O'
                tokens = tokenizer_data.tokenize(word)

                if len(tokens) + len(x) >= sequence_len:
                    break
                elif word == '<UNK>':
                    x.append(TOKEN_IDX[token_style]['UNK'])
                    y.append(punctuation_dict[punc])
                    y_mask.append(1)
                    idx += 1
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer_data.convert_tokens_to_ids(tokens[i]))
                        y.append(0)
                        y_mask.append(0)

                    if len(tokens) > 0:
                        x.append(tokenizer_data.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x.append(TOKEN_IDX[token_style]['UNK'])

                    y.append(punctuation_dict[punc])
                    y_mask.append(1)
                    idx += 1

            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            y.append(0)
            y_mask.append(1)
            if len(x) < sequence_len:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
                y = y + [0 for _ in range(sequence_len - len(y))]
                y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]

            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
            data_items.append([x, y, attn_mask, y_mask])
        return data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]
        y_mask = self.data[index][3]

        if self.is_train and self.augment_rate > 0:
            x, y, attn_mask, y_mask = self._augment(x, y, y_mask)

        x = torch.tensor(x)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        y_mask = torch.tensor(y_mask)

        return x, y, attn_mask, y_mask
