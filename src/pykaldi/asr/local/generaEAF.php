<?php
    //Entrada de parámetros
    //1: Ruta ó ficheiro de segmentos que ten a seguinte estructura por liña: proba-000000-001190 proba 0.00 11.90
    //2: Ruta ó ficheiro fruto do recoñecemento que ten a estructura: proba-000000-001190 1 0.17 0.49 asunto 0.64
    //3: Ficheiro onde se desexa deixar a saida (sen extensión, porque vai ser o mesmo nome que o ficheiro de audio)
    //Engádense o ficheiro de configuración da aplicación 
    include("config.php");
    
    //Comprobación do número de parámetros de entrada
    if ($argc!=4) {
        echo "Produciuse un erro á hora de xerar o eaf";
    }
    
    //Apertura do ficheiro de segmentos para ir obtendo o identificador do segmento, así como o seu tempo inicial
    $f_segmentos=fopen($argv[1],"r");
    
    //Apertura do ficheiro de recoñecemento para ir recorrendo as liñas do recoñecido
    $f_reco=fopen($argv[2],"r");
    
    //Apertura do ficheiro de saida, que vai conter a transformación do recoñecido a formato eaf
    $f_saida=fopen($argv[3].".eaf","w");
    
    //Definición de variables para creación da cabeceira do eaf
    $audio_file_path="./audios/audio_path.wav";
    $audio_file_type="audio/wav";
		
    /* annotation parameters */ //estos parámetros depende de como se pasen y de la nomenclatura    
    $autor_name="AUTOMATIC";
//     $date=date(DATE_ATOM,getdate());
    $date=date(DATE_RFC2822);
    $annotator_name="AUTOMATIC";
    $linguistic_type_ref="default_lt";
    $participant="AUTOMATIC";
    
    //Nota:Xa se podería ir escribindo no ficheiro en realidade -> menos consumo de memoria    
    //Definición do elemento annotation_document
    fwrite($f_saida,utf8_encode("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
//     $document="<?xml version=\"1.0\" encoding=\"UTF-8\"?\>\n";
    fwrite($f_saida,utf8_encode("<ANNOTATION_DOCUMENT AUTHOR=\"$autor_name\" DATE=\"$date\" FORMAT=\"2.7\" VERSION=\"2.7\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://www.mpi.nl/tools/elan/EAFv2.7.xsd\">\n"));
//     $document.="<ANNOTATION_DOCUMENT AUTHOR=\"$autor_name\" DATE=\"$date\" FORMAT=\"2.7\" VERSION=\"2.7\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://www.mpi.nl/tools/elan/EAFv2.7.xsd\">\n";
    
    //Definición do elemento HEADER só hai un por documento 
    fwrite($f_saida,utf8_encode("\t<HEADER MEDIA_FILE=\"\" TIME_UNITS=\"milliseconds\">\n"));
//     $header="\t<HEADER MEDIA_FILE=\"\" TIME_UNITS=\"milliseconds\">\n";//Por defecto as unidades son milisegundos
    fwrite($f_saida,utf8_encode("\t\t<MEDIA_DESCRIPTOR MEDIA_URL=\"./".$argv[3].".wav\" MIME_TYPE=\"$audio_file_type\"/>\n"));
//     $header.="\t\t<MEDIA_DESCRIPTOR MEDIA_URL=\"./".$argv[3].".wav\" MIME_TYPE=\"$audio_file_type\"/>\n";
    fwrite($f_saida,utf8_encode("\t</HEADER>\n"));
//     $header.="\t</HEADER>\n";
    
    //Definicón do elemento que vai conter os tempos
    fwrite($f_saida,utf8_encode("\t<TIME_ORDER>\n"));
//     $timeorder="\t<TIME_ORDER>\n";
    
    //Lese a liña do ficheiro de segmentos
    if (!feof($f_segmentos)) {
        $l_segmentos=fgets($f_segmentos);
        //Divídese a liña lida do ficheiro de segmentos por espacios en branco -> despois vaise utilizar a posición 0 (id_segmento) e a posición 1 (tempo inicial do segmento dentro do audio global)
        $v_l_segmentos=explode(" ",$l_segmentos);
    }
    
    
    
    //Inicialización da variable que vai conter o vector de recoñecemento, desta forma somos capaces de saber se é a primeira vez ou non, porque hai que ter a información da liña anterior antes 
    $v_l_reco="";
    
    //Inicialización da variable que vai conter a cadea coas palabras
    $cadena="";
    
    //Inicialización do tier que vai conter a segmentación en palabras
    $tier_palabra="\t<TIER ANNOTATOR=\"Automatic\" LINGUISTIC_TYPE_REF=\"default-lt\" PARTICIPANT=\"\" TIER_ID=\"Palabra\">\n";
    //Inicialización do tier que vai conter a segmentación en segmentos
    $tier_segmento="\t<TIER ANNOTATOR=\"Automatic\" LINGUISTIC_TYPE_REF=\"default-lt\" PARTICIPANT=\"\" TIER_ID=\"Segmento\">\n";
    
    //Variable que se vai utilizar para crear os identificadores dos tempos nos time_slot
    $contador_slot=1;
    
    //Variable que se vai utilizar para crear o identificador das anotacións
    $contador_anot=1;
    
    //Variable que vai conter o identificador do time_slot de inicio do segmento
    $id_tini_segmento=1;
    
    $t_ini_segmento=$v_l_segmentos[2];
    //Bloque para recorrer o ficheiro que contén o recoñecemento
    while (!feof($f_reco)) {
        //Lese a fila do recoñecido
        $l_reco=fgets($f_reco);
        //Se a liña nn está vacía -> engádese ó ficheiro os identificadores dos tempos cos valores e enchénse os arrays que conteñen as anotacións a nivel de palabra e de segmento
        if (!ctype_space($l_reco) && $l_reco!="") {
    //         //Obtención do tempo final do ficheiro de recoñecemento
    //         if ($v_l_reco!="") $t_fin=$v_l_reco[2] + $v_l_reco[3];
            
            //Divídese a liña lida do ficheiro de recoñecemento por espacios en branco -> despois vaise utilizar a posición 0 (id_segmento) e a posición 1 (tempo inicial do segmento dentro do audio global)
            $v_l_reco=explode(" ",$l_reco);
//               echo "Liña_reco:".$l_reco;
            
            
            
            //Prodúcese un cambio de segmento -> hai que ler o ficheiro de segmentos
            if ($v_l_segmentos[0]!=$v_l_reco[0]) {
//             echo "SON DISTINTOS:";
//             echo "Segmento: ".$v_l_segmentos[0]." Reco:".$v_l_reco[0];
                //Lese a liña do ficheiro de segmentos
                if (!feof($f_segmentos)) {
                    $l_segmentos=fgets($f_segmentos);
//                     echo "\nsegmentos ".$l_segmentos."algo";
                    //Divídese a liña lida do ficheiro de segmentos por espacios en branco -> despois vaise utilizar a posición 0 (id_segmento) e a posición 1 (tempo inicial do segmento dentro do audio global)
                    if (!ctype_space($l_segmentos)&& $l_segmentos!="") $v_l_segmentos=explode(" ",$l_segmentos);
                }
                //Transformación do tempo inicial e final do segmento a milisegundos
//                 $t_ini_segmento=$v_l_segmentos[2]*1000;
                $t_ini_segmento=$v_l_segmentos[2];
//                echo "\nt_ini_segmento ".$t_ini_segmento;
//                echo "\nv_l_reco[2] ".$v_l_reco[2];

/*                $t_fin_segmento=trim($v_l_segmentos[3])*1000; //Nota: Hai que aplicar trim porque contén o salto de línea do ficheiro */             
                
                if ($cadena!=""){
                //Creación da anotación que contén o segmento
                    $tier_segmento.="\t\t<ANNOTATION>\n";
                    $tier_segmento.="\t\t\t<ALIGNABLE_ANNOTATION ANNOTATION_ID=\"a".$contador_anot++."\" TIME_SLOT_REF1=\"ts$id_tini_segmento\" TIME_SLOT_REF2=\"ts".($contador_slot-1)."\">\n";
                    $tier_segmento.="\t\t\t \t<ANNOTATION_VALUE>$cadena</ANNOTATION_VALUE>\n";
                    $tier_segmento.="\t\t\t</ALIGNABLE_ANNOTATION>\n";
                    $tier_segmento.="\t\t</ANNOTATION>\n";
                }
                
                
                
                //Obtención do tempo final da palabra (t_ini_segmento + t_ini + duracion)
//                 $t_fin_palabra=($t_ini_segmento + $v_l_reco[2] + $v_l_reco[3])*1000;
                //Almacenamento do identificador do inicio do próximo segmento
                $id_tini_segmento=$contador_slot;      
                
                //Escríbese no ficheiro o time_slot correspondente ó tempo inicial da palabra
                fwrite($f_saida,utf8_encode("\t\t<TIME_SLOT TIME_SLOT_ID=\"ts".$contador_slot++."\" TIME_VALUE=\"".(($t_ini_segmento + $v_l_reco[2])*1000)."\"/>\n"));
                //Creación da anotación que contén a palabra
                $tier_palabra.="\t\t<ANNOTATION>\n";
		$tier_palabra.="\t\t\t<ALIGNABLE_ANNOTATION ANNOTATION_ID=\"a".$contador_anot++."\" TIME_SLOT_REF1=\"ts".($contador_slot-1)."\" TIME_SLOT_REF2=\"ts".$contador_slot."\">\n";
		$tier_palabra.="\t\t\t \t<ANNOTATION_VALUE>".$v_l_reco[4]."</ANNOTATION_VALUE>\n";
		$tier_palabra.="\t\t\t</ALIGNABLE_ANNOTATION>\n";
                $tier_palabra.="\t\t</ANNOTATION>\n";
                
                          
                
                //Escríbese no ficheiro o time_slot correspondente ó tempo final da palabra
                fwrite($f_saida,utf8_encode("\t\t<TIME_SLOT TIME_SLOT_ID=\"ts".$contador_slot++."\" TIME_VALUE=\"".(($t_ini_segmento + $v_l_reco[2] + $v_l_reco[3])*1000)."\"/>\n"));
                
                $cadena=$v_l_reco[4];

                
                
                
            } else {
                //Non se produce un cambio de segmento
                if ($cadena!="") $cadena.=" ".$v_l_reco[4];
                else $cadena=$v_l_reco[4];
                
                //Escríbese no ficheiro o time_slot correspondente ó tempo inicial da palabra
                fwrite($f_saida,utf8_encode("\t\t<TIME_SLOT TIME_SLOT_ID=\"ts".$contador_slot++."\" TIME_VALUE=\"".(($t_ini_segmento + $v_l_reco[2])*1000)."\"/>\n"));             
                //Creación da anotación que contén a palabra
                $tier_palabra.="\t\t<ANNOTATION>\n";
		$tier_palabra.="\t\t\t<ALIGNABLE_ANNOTATION ANNOTATION_ID=\"a".$contador_anot++."\" TIME_SLOT_REF1=\"ts".($contador_slot-1)."\" TIME_SLOT_REF2=\"ts".$contador_slot."\">\n";
		$tier_palabra.="\t\t\t \t<ANNOTATION_VALUE>".$v_l_reco[4]."</ANNOTATION_VALUE>\n";
		$tier_palabra.="\t\t\t</ALIGNABLE_ANNOTATION>\n";
                $tier_palabra.="\t\t</ANNOTATION>\n";
                //Escríbese no ficheiro o time_slot correspondente ó tempo final da palabra
                fwrite($f_saida,utf8_encode("\t\t<TIME_SLOT TIME_SLOT_ID=\"ts".$contador_slot++."\" TIME_VALUE=\"".(($t_ini_segmento + $v_l_reco[2] + $v_l_reco[3])*1000)."\"/>\n"));
            }
        }
        
    }
    
    //Péchase o elemento de tempos
    fwrite($f_saida,utf8_encode("\t</TIME_ORDER>\n"));
    
    //Engádese o tier de palabra
    fwrite($f_saida,utf8_encode($tier_palabra));
    
    //Péchase o tier de palabra
    fwrite($f_saida,utf8_encode("\t</TIER>\n"));
    
    //Engádese o tier de segmento
    fwrite($f_saida,utf8_encode($tier_segmento));
    
    //Ainda queda un segmento máis
    //Creación da anotación que contén o segmento
    $tier_segmento="\t\t<ANNOTATION>\n";
    $tier_segmento.="\t\t\t<ALIGNABLE_ANNOTATION ANNOTATION_ID=\"a".$contador_anot++."\" TIME_SLOT_REF1=\"ts$id_tini_segmento\" TIME_SLOT_REF2=\"ts".($contador_slot-1)."\">\n";
    $tier_segmento.="\t\t\t \t<ANNOTATION_VALUE>$cadena</ANNOTATION_VALUE>\n";
    $tier_segmento.="\t\t\t</ALIGNABLE_ANNOTATION>\n";
    $tier_segmento.="\t\t</ANNOTATION>\n";
    //Engádese o tier de segmento
    fwrite($f_saida,utf8_encode($tier_segmento));
    //Péchase o tier de segmento
    fwrite($f_saida,"\t</TIER>\n");
    
    //Engádense as partes finais do documento (entendo que son estáticas, pero non estou segura)
    fwrite($f_saida,utf8_encode("\t<LINGUISTIC_TYPE GRAPHIC_REFERENCES=\"false\" LINGUISTIC_TYPE_ID=\"default-lt\" TIME_ALIGNABLE=\"true\"/>\n"));
    fwrite($f_saida,utf8_encode("\t<CONSTRAINT DESCRIPTION=\"Time subdivision of parent annotation's time interval, no time gaps allowed within this interval\" STEREOTYPE=\"Time_Subdivision\"/>\n"));
    fwrite($f_saida,utf8_encode("\t<CONSTRAINT DESCRIPTION=\"Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered\" STEREOTYPE=\"Symbolic_Subdivision\"/>\n"));
    fwrite($f_saida,utf8_encode("\t<CONSTRAINT DESCRIPTION=\"1-1 association with a parent annotation\" STEREOTYPE=\"Symbolic_Association\"/>\n"));
    fwrite($f_saida,utf8_encode("\t<CONSTRAINT DESCRIPTION=\"Time alignable annotations within the parent annotation's time interval, gaps are allowed\" STEREOTYPE=\"Included_In\"/>\n"));
    
    //Péchase o annotation documente
    fwrite($f_saida,utf8_encode("</ANNOTATION_DOCUMENT>\n"));
    
    
    
    //Péchanse os ficheiros que foron abertos    
    fclose($f_segmentos);
    fclose($f_reco);
    fclose($f_saida);


    

?>
