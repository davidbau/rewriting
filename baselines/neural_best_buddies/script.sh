while true;
do 

for NUM in $(seq 0 100);
do

python3 main.py --source ../../utils/samples/clean --target images/child.png --imgnum ${NUM} --k_final 10 --k_per_level 10 --results_dir child --fast
python3 main.py --source ../../utils/samples/clean --target images/smile.png --imgnum ${NUM} --k_final 10 --k_per_level 10 --results_dir smile --fast
python3 main.py --source ../../utils/samples/domes --target images/dome.png --imgnum ${NUM} --k_final 10 --k_per_level 10 --results_dir domes --fast
python3 main.py --source ../../utils/samples/church --target images/dome.png --imgnum ${NUM} --k_final 10 --k_per_level 10 --results_dir church_all --fast

done

sleep 5

done
