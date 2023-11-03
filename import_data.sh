# Loads in all relevant datasets
source ../../miniconda/bin/activate
conda activate ./venv

load() {
  local name="$1"
  shift
  echo
  echo
  echo "Importing $name"
  python load_dataset.py $name "./datasets/csv/$name.csv" --output ./datasets/binary/ \
    --validate_percent 0.2 --test_percent 0.3 --seed 1337 "$@"
}

load abalone --target 'Rings' --categorical 'Sex' \
  --numeric 'Length' 'Diameter' 'Height' "Whole weight" "Shucked weight" "Viscera weight" "Shell weight"
load delta_ailerons --target 'Sa' \
  --numeric 'RollRate' 'PitchRate' 'currPitch' 'currRoll' 'diffRollRate'
load elevators --target 'Goal' \
  --numeric 'climbRate' 'Sgz' 'p' 'q' 'curRoll' 'absRoll' 'diffClb' 'diffRollRate' 'diffDiffClb' \
    'SaTime1' 'SaTime2' 'SaTime3' 'SaTime4' 'diffSaTime1' 'diffSaTime2' 'diffSaTime3' 'diffSaTime4' 'Sa'
load insurance --target 'charges' --categorical 'smoker' 'region' 'sex' \
  --numeric 'age'  'bmi' 'children'