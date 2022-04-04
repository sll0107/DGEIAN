# DGEIAN
some code is from https://github.com/NJUNLP/GTS, and thanks them

## Data
copy datasets from https://github.com/NJUNLP/GTS/tree/main/data and https://github.com/xuuuluuu/SemEval-Triplet-data

## Requirements
See requirement.txt or Pipfile for details

## Usage
- ### Training
```
python main.py --mode train --dataset res14
```
The best model will be saved in the folder "savemodel/".

- ### Testing
```
python main.py --mode test --dataset res14
```

