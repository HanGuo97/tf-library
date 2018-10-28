##################### TUNER

gArg1=${gridArg1}
gArg2=${gridArg2}
gArg3=${gridArg3}
gArg4=${gridArg4}
gArg5=${gridArg5}
gArg6=${gridArg6}

bArg1=${bayesianArg1}
bArg2=${bayesianArg2}
bArg3=${bayesianArg3}
bArg4=${bayesianArg4}
bArg5=${bayesianArg5}

echo "gArg1=${gArg1} gArg2=${gArg2} gArg3=${gArg3} gArg4=${gArg4} gArg5=${gArg5} gArg6=${gArg6}"

echo $RANDOM > "TunerTest/\
${gArg1}_${gArg2}_${gArg3}_${gArg4}_${gArg5}_${gArg6}_\
${bArg1}_${bArg2}_${bArg3}_${bArg4}_${bArg5}.observ"