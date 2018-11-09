##################### TUNER

gArg1=${gridArg1}
gArg2=${gridArg2}
bArg1=${bayesianArg1}
bArg2=${bayesianArg2}

echo "gArg1=${gArg1} gArg2=${gArg2}"

echo $RANDOM > "TunerTest/${gArg1}_${gArg2}_${bArg1}_${bArg2}.observ"