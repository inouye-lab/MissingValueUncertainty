# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

experiment() {
  local name="$1"
  local date="$2"
  echo
  echo
  echo "Running $name"
  python missing_experiments.py $name --dataset "./datasets/binary/$name.pklz" \
    --regressor "./models/ridge/$name-$date.pklz" --output ./results/ridge_feature_impact/ \
    --feature_impact --mc_samples 10 100 1000 --seed 1337
}

experiment abalone "20231109-003846"
experiment delta_ailerons "20231109-003849"
experiment insurance "20231109-003853"
experiment elevators "20231109-003851"