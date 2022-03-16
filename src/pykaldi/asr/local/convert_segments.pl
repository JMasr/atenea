#!/usr/bin/perl

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This takes as standard input a ctm file that's "relative to the utterance",
# i.e. times are measured relative to the beginning of the segments, and it
# uses a "segments" file (format:
# utterance-id recording-id start-time end-time
# ) and a "reco2file_and_channel" file (format:
# recording-id basename-of-file

$skip_unknown=undef;
if ( $ARGV[0] eq "--skip-unknown" ) {
  $skip_unknown=1;
  shift @ARGV;
}

if (@ARGV < 1 || @ARGV > 2) {
  print STDERR "Usage: convert_segments.pl <segments-file> [<new-segments-file>] > real-new-segments-file\n";
  exit(1);
}

$segments = shift @ARGV;
$reco2file_and_channel = shift @ARGV;

open(S, "<$segments") || die "opening segments file $segments";
while(<S>) {
  @A = split(" ", $_);
  @A == 4 || die "Bad line in segments file: $_";
  ($utt, $recording_id, $begin_time, $end_time) = @A;
  $utt2reco{$utt} = $recording_id;
  $begin{$utt} = $begin_time;
  $end{$utt} = $end_time;
}
close(S);


# Now process the new segments file, which is either the standard input or the second
# command-line argument.
$num_done = 0;
while(<>) {
  @A= split(" ", $_);
  @A == 4 || die "Bad line in segments file: $_";
  ($utt, $recording_id, $begin_time, $end_time) = @A;
  $reco = $utt2reco{$recording_id};
  if (!defined $reco) { 
      next if defined $skip_unknown;
      die "Utterance-id $recording_id not defined in segments file $segments"; 
  }
  $b = $begin{$recording_id};
  $e = $end{$recording_id};
  $ubegin_r = $begin_time + $b; # Make it relative to beginning of the recording.
  $ubegin_r = sprintf("%.3f", $ubegin_r);
  $uend_r = $end_time + $b;
  $uend_r = sprintf("%.3f", $uend_r);
  $line = sprintf("%s-%.8i-%.8i %s %.3f %.3f\n", $recording_id,1000*$ubegin_r,1000*$uend_r,$reco,$ubegin_r,$uend_r); 
  if ($uend_r > $e + 0.01) {
    print STDERR "Warning: utterance appears to be past end of recording; line is $line";
  }
  print $line; # goes to stdout.
  $num_done++;
}

if ($num_done == 0) { exit 1; } else { exit 0; }

__END__
