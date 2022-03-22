#!/usr/bin/perl

#use Switch;

($entrada,$salida) = @ARGV;

open(ENTRADA,"$entrada") || die "No se puede abrir $entrada\n";

@rttm = <ENTRADA>;

close(ENTRADA);

@inicios = ();
@finales = ();
@etiquetas = ();

$tf = 0;
for($i = 0; $i <= $#rttm; $i++) {

  ($nada1,$nombreFichero,$nada3,$inicio,$duracion,$nada4,$nada5,$etiqueta,$nada6) = split(/\s+/,$rttm[$i]);
	#($utt,$nombreFichero,$inicio,$final) = split(/\s/,$rttm[$i]);
	push(@inicios,$inicio);
  $final=$inicio+$duracion;
	push(@finales,$final);
	push(@duraciones,$duracion);
	push(@utts,$utt);
	push(@etiquetas,$etiqueta);
	if ($final > $tf) {
		$tf = $final;
	}
}

$inicio1 = $inicios[0];
$fin1 = $finales[0];

for($i = 0; $i <= $#inicios-1; $i++) {
	$inicio1 = $inicios[$i];
	$fin1 = $finales[$i];
	$duracion1 = $duraciones[$i];
	$inicio2 = $inicios[$i+1];
	$fin2 = $finales[$i+1];	
  $duracion2 = $duraciones[$i+1];

  if ($fin1 >= $inicio2 ) {
    $finales[$i] = $inicio2;
  }
}


open(SALIDA,">$salida") || die "No se puede crear $salida\n";
for($i = 0; $i <= $#inicios; $i++) {
  $duracion=$finales[$i]-$inicios[$i];
  printf SALIDA "SPEAKER %s 1 %.6f %.6f <NA> <NA> %s <NA>\n",$nombreFichero,$inicios[$i],$duracion,$etiquetas[$i];
}

close(SALIDA);
