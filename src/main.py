from model import Pipeline
from argparser import parse_arguments

if __name__ == '__main__':
    args = parse_arguments()
    rich_t = Pipeline(args)
    rich_t.apply_diarization()
    rich_t.apply_asr()
    rich_t.apply_acpr()
    rich_t.make_srt()
