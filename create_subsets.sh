ls

for lg in $@;
do head --bytes 50000000 $lg/${lg}_shuf_part_1.txt > $lg/${lg}_shuf_part_0.05.txt
head --bytes 100000000 $lg/${lg}_shuf_part_1.txt > $lg/${lg}_shuf_part_0.1.txt
head --bytes 500000000 $lg/${lg}_shuf_part_1.txt > $lg/${lg}_shuf_part_0.5.txt
done
