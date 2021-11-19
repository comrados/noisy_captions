probs=(0.0 0.1 0.2 0.3 0.4 0.5)
noise=("normal" "exp" "dis" "ones")


for j in "${noise[@]}"
  do
    for i in "${probs[@]}"
      do
        echo
        /home/george/Code/venvs/new_venv/bin/python /home/george/Code/noisy_captions/main.py --dataset rsicd --noise-wrong-caption $i --clean-captions 0.2 --noise-weights $j
      done
  done