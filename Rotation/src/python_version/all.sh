

for i in `seq 15 14`;
do
    cl info ^$i >> out100
    cl cat ^$i/stdout >> out100
done   
