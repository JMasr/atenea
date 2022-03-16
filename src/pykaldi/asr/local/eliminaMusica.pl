#!/usr/bin/perl

($fSegmentosEntrada,$fProbClases,$fSegmentosSalida) = @ARGV;

open(fEntrada,$fSegmentosEntrada) || die "No se puede abrir $fSegmentosEntrada\n";
open(fClases,$fProbClases) || die "No se puede abrir $fProbClases\n";
open(fSalida,">$fSegmentosSalida") || die "No se puede crear $fSegmentosSalida\n";

@datosEntrada = <fEntrada>;
@probabilidades = <fClases>;

close(fEntrada);
close(fClases);

for($i = 0; $i < @datosEntrada; $i++) {
    @probs = split(/\s/,$probabilidades[$i]);
#    printf "@probs\n";
#    $_ = <STDIN>;
    $maximo=$probs[0];
    $indice=0;
    for($j = 1; $j < @probs; $j++) {
	if($probs[$j] > $maximo) {
	    $maximo=$probs[$j];
	    $indice=$j;
	}
    }
#    printf "$indice\n";
    if($indice != 1) {
	printf fSalida "$datosEntrada[$i]";
    }
}

close(fSalida);
