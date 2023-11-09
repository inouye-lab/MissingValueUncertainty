# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

learn() {
  local name="$1"
  echo
  echo
  echo "Learning $name"
  python learn_regressor_ridge.py $name "./datasets/binary/$name.pklz" --output ./models/ridge/ \
    --params '{"solver": "auto"}' --seed 1337
}

learn abalone
learn delta_ailerons
learn elevators
learn insurance