import os

from model import RichTranscription
from argparser import parse_arguments

args = parse_arguments()
rich_t = RichTranscription(args)
print(rich_t.speech_timestamps)
