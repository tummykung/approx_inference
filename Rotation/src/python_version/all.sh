

for i in `seq 9 1`;
do
    cl info ^$i >> out5
    cl cat ^$i/stdout >> out5
done   
