#!/bin/bash
npts=0
for file in "$@"
do 
    npt=$(head -n 1 "$file")
    npts=$((npts+npt))
done

fname=${file##*/}
prefix=${fname%%.*}
exportname=${prefix##*/}.xyz

echo "Merging files ${prefix}.*.xyz into $exportname."

# write first line (number of points)
echo $npts > ${exportname}

# write second line (expanded xyz format)
head -n 2 $file | tail -n 1 >> ${exportname}

echo -n "Attaching file into ${exportname}"
for file in "$@"
do
    echo -n "...${file}"
    $(tail -n +3 $file >> ${exportname})
done
echo ""
