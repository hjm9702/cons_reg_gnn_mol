

cd data
python get_molnet.py
cd ..

for split in 3 2 1 0
do 
    for d_name in bace bbbp clintox sider tox21 toxcast hiv muv
    do        
        for seed in {27407..27411} 
        do
            for cvid in {0..4}
            do

                python main.py --seed $seed --cvid $cvid --d_name $d_name --split $split
            done
        done
    done
done
