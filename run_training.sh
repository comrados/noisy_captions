probs=(0.0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for i in "${probs[@]}"
do
  echo
	/home/george/Code/venvs/new_venv/bin/python /home/george/Code/noisy_captions/main.py --dataset $1 --noise-wrong-caption $i
done
