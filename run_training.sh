probs=(0.0 0.1 0.2 0.3 0.4 0.5)
cleans=(0.1 0.2 0.3)


for j in "${cleans[@]}"
  do
    for i in "${probs[@]}"
      do
        echo
        /home/george/Code/venvs/new_venv/bin/python /home/george/Code/noisy_captions/main.py --dataset $1 --noise-wrong-caption $i --clean-captions $j
      done
  done