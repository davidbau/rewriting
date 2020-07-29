

#!/bin/bash


while true;
do

for DATASET in celebhq
do

for GANTYPE in proggan 
do

for LAYERNUM in 3 4 5 6 7 8 9 
do

for SIZE in 1 2 4 8 16 32 64
do 

python3 fid_image_patches.py --layernum ${LAYERNUM} --crop_size ${SIZE} --model ${GANTYPE} --dataset ${DATASET} --nimgs 50000

done
done
done
done

sleep 5
done




