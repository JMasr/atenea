from model import Pipeline
from argparser import parse_arguments


args = parse_arguments()
rich_t = Pipeline(args)
rich_t.apply_vad()
rich_t.audio.show_vad()
