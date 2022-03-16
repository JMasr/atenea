#! /usr/bin/env python

# Copyright 2014  Johns Hopkins University (Authors: Daniel Povey)
#           2014  Vijayaditya Peddinti
#           2016  Vimal Manohar
#           2019  Laura
# Apache 2.0.


from __future__ import print_function
from __future__ import division
import argparse
import collections
import logging

from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s [%(pathname)s:%(lineno)s - '
    '%(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    """gets command line arguments"""

    usage = """ Python script to resolve offsets in ctms """
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('offsets', type=argparse.FileType('r'),
                        help='use offsets file')
    parser.add_argument('ctm_in', type=argparse.FileType('r'),
                        help='input_ctm_file')
    parser.add_argument('ctm_out', type=argparse.FileType('w'),
                        help='output_ctm_file')
    parser.add_argument('--verbose', type=int, default=0,
                        help="Higher value for more verbose logging.")
    args = parser.parse_args()

    if args.verbose > 2:
        logger.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)

    return args


def read_segments(segments_file):
    """Read from segments and returns two dictionaries,
    {utterance-id: (recording_id, start_time, end_time)}
    {recording_id: list-of-utterances}
    """
    segments = {}
    reco2utt = defaultdict(list)

    num_lines = 0
    for line in segments_file:
        num_lines += 1
        parts = line.strip().split()
        assert len(parts) in [4, 5]
        segments[parts[0]] = (parts[1], float(parts[2]), float(parts[3]))
        reco2utt[parts[1]].append(parts[0])

    logger.info("Read %d lines from segments file %s",
                num_lines, segments_file.name)
    segments_file.close()

    return segments, reco2utt


def read_offsets(offsets_file):
    """Read from offsets and returns two dictionaries,
    {utterance-id: (recording_id, offset_time)}
    {recording_id: list-of-utterances}
    """
    offsets = {}
    reco2utt = defaultdict(list)

    num_lines = 0
    for line in offsets_file:
        num_lines += 1
        parts = line.strip().split()
        assert len(parts) in [3, 4]
        offsets[parts[0]] = (parts[1], float(parts[2]))
        reco2utt[parts[1]].append(parts[0])

    logger.info("Read %d lines from offsets file %s",
                num_lines, offsets_file.name)
    offsets_file.close()

    return offsets, reco2utt



def modify_ctm(ctm_file, offsets, out_file):
    """Read CTM from ctm_file into a dictionary of values indexed by the
    recording and sum offsets to it. Then write it.
    It is assumed to be sorted by the recording-id and utterance-id.
    Returns a dictionary {recording : ctm_lines}
        where ctm_lines is a list of lines of CTM corresponding to the
        utterances in the recording.
        The format is as follows:
        [[(utteranceA, channelA, start_time1, duration1, hyp_word1, conf1),
          (utteranceA, channelA, start_time2, duration2, hyp_word2, conf2),
          ...
          (utteranceA, channelA, start_timeN, durationN, hyp_wordN, confN)],
         [(utteranceB, channelB, start_time1, duration1, hyp_word1, conf1),
          (utteranceB, channelB, start_time2, duration2, hyp_word2, conf2),
          ...],
         ...
         [...
          (utteranceZ, channelZ, start_timeN, durationN, hyp_wordN, confN)]
        ]
    """
    ctms = {}

    num_lines = 0
    for line in ctm_file:
        num_lines += 1
        parts = line.split()

        utt = parts[0]
        reco = offsets[utt][0]
        vad_offset = offsets[utt][1]
        if (reco, utt) not in ctms:
            ctms[(reco, utt)] = []

        start_time = float(parts[2])+float(vad_offset)
        ctms[(reco, utt)].append([parts[0], parts[1], start_time, float(parts[3])] + parts[4:])
        newline = "{0} {1} {2} {3} {4}".format(reco, parts[1], start_time, parts[3]," ".join(parts[4:]))
        print(newline, file=out_file)

    logger.info("Read %d lines from CTM %s", num_lines, ctm_file.name)

    ctm_file.close()
    return ctms



def run(args):
    """this method does everything in this script"""
    offsets, reco2utt = read_offsets(args.offsets)
    ctms = modify_ctm(args.ctm_in, offsets, args.ctm_out)

    #write_ctm(ctms, args.ctm_out)

    args.ctm_out.close()
    logger.info("Wrote CTM for %d recordings.", len(ctms))


def main():
    """The main function which parses arguments and call run()."""
    args = get_args()
    try:
        run(args)
    except:
        logger.error("Failed to resolve overlaps", exc_info=True)
        raise SystemExit(1)
    finally:
        try:
            for f in [args.offsets, args.ctm_in, args.ctm_out]:
                if f is not None:
                    f.close()
        except IOError:
            logger.error("Could not close some files. "
                         "Disk error or broken pipes?")
            raise
        except UnboundLocalError:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

