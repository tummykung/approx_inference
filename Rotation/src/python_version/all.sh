

for i in `seq 13 1`;
do
    cl info ^$i >> out4
    cl cat ^$i/stdout >> out4
done   