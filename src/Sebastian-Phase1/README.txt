Enter venv (uses python 3.11)
source venv311/bin/activate.fish

Use this command and modify per needs

python train_phase1_gemstonesv3.py \
                --arch resnet50 \
                --dataset_root "/home/seb/Documents/CSC-481/gem-sorter/datasets/Phase1-Combined-Dataset/Combined-P1-Dataset" \
                --results_dir latest_results \
                --epochs 1 \
                --batch_size 32
