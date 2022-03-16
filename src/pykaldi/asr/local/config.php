<?php

/* Database access information */
// $database="intras";
$database="subtitulado";
$host="localhost";
$user="seix_usu_sub";
$password="/f:mD@i4AF0Mm*EA})Y%4o~VV";

//Variable que contén a ruta onde se atopa instalada a aplicación
$ruta_aplicacion="/var/www/html/Subtitulado_probas";


//Variable que vai conter a ruta onde se desexan almacenar os audios  (é relativa á ruta da aplicación, a ruta definitiva sería $ruta_aplicacion/$ruta_grabaciones) engádese tmp porque é a carpeta que se creou con todos os persmisos. OLLO!! Se se modifica a  ruta hai que ter coidado que teña permisos para poder crear carpetas en caso de que esta non os teña ou se existe a carpeta que teña permisos para poder escribir en ela
$ruta_grabaciones="tmp/audio";

$base_aplicacion="Subtitulado_probas";

?>
