probs=(0.05 0.1 0.2 0.3 0.4 0.5)


for i in "${probs[@]}"
  do
    echo
    /home/george/Code/venvs/new_venv/bin/python /home/george/Code/noisy_captions/main.py --dataset rsicd --noise-wrong-caption $i --clean-captions 0.2 --noise-weights ones --tag default_baseline --clean-epochs 0
  done

for i in "${probs[@]}"
  do
    echo
    /home/george/Code/venvs/new_venv/bin/python /home/george/Code/noisy_captions/main.py --dataset ucm --noise-wrong-caption $i --clean-captions 0.3 --noise-weights ones --tag default_baseline --clean-epochs 0
  done
