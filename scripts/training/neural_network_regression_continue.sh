# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

learn() {
  local name="$1"
  local date="$2"
  shift
  shift
  echo
  echo
  echo "Learning $name"
  python learn_neural_network.py $name "./datasets/binary/$name.pklz" --output ./models/nn/ \
    --input "./models/nn/$name-$date.pklz" --seed 1234 "$@"
}

learn abalone "20231116-220802" --training_iterations 1000
learn delta_ailerons "20231116-221117" --training_iterations 1000 --patience 10
learn elevators "20231116-221149" --training_iterations 1000 --patience 10
learn insurance "20231116-221525" --training_iterations 2000