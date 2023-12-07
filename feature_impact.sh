# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

experiment() {
  local name="$1"
  local regressor="$2"
  local date="$3"
  echo
  echo
  echo "Running $name"
  python missing_experiments.py $name --regressor "./models/$regressor/$name-$date.pklz" \
    --output "./results/${regressor}_feature_impact/" --feature_impact --mc_samples 10 100 1000 --seed 1337
  python missing_experiments.py $name --regressor "./models/$regressor/$name-$date.pklz" \
    --output "./results/${regressor}_inverted_feature_impact/" --inverted_feature_impact --mc_samples 10 100 1000 --seed 1337
}

#experiment abalone ridge "20231109-003846"
#experiment delta_ailerons ridge "20231109-003849"
#experiment insurance ridge "20231109-003853"
#experiment elevators ridge "20231109-003851"

experiment abalone nn "20231116-220802"
experiment delta_ailerons nn "20231116-221117"
experiment insurance nn "20231116-221525"
experiment elevators nn "20231116-221149"