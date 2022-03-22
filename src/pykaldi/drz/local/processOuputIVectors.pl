#!/usr/bin/perl

#use Switch;

($segments,$recofile,$outrttm) = @ARGV;

open(S,"<$segments") || die "Can't open segments file: $segments\n";
while(<S>) {
  @A = split(" ", $_);
  @A == 4 || die "Bad line in segments file: $_";
  ($utt, $recording_id, $begin_time, $end_time) = @A;
  $utt2reco{$utt} = $recording_id;
  $begin{$utt} = $begin_time;
  $end{$utt} = $end_time;
}
close(S);

open(R, "<$recofile") || die "Can't open recofile file $recofile";
open(RTTM, ">$outrttm") || die "Can't open recofile file $outrttm";
while(<R>) {
  @A = split(" ", $_);
  ($utt, $label) = @A;

  $reco = $utt2reco{$utt};
  $wbegin = $begin{$utt};
  $wend = $end{$utt};
  $wlen = $wend - $wbegin;
  printf RTTM "SPEAKER %s 1 %.2f %.2f <NA> <NA> %s <NA>\n", $reco, $wbegin, $wlen, $label;
 
}

close(R);
close(RTTM);


