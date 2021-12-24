from model import RichTranscription
from argparser import parse_arguments

args = parse_arguments()
rich_t = RichTranscription(args)
print(rich_t.apply_vad())
rich_t.save_speech_segments()
