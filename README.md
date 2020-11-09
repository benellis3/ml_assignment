# ml_assignment
Assignment for the AIMS CDT ML course


# Reproducing results

Install the relevant packages in a virtualenv by running `pip install -r requirements.txt`

## Benchmarking
run `python benchmark.py`. These will vary depending on your machine. The results quoted were obtained on my Macbook Pro.

## IRIS dataset results
run `python main.py --dataset iris --epochs 500 --lr 0.001`. The run used in the report has its logs in the figures/ subfolder.

## KMNIST results
run `python main.py --dataset kmnist --epochs 15 --lr 0.01`. The run used in the report has its logs in the figures/ subfolder also.

To reproduce the grid simply run `python main.py --dataset kmnist --plot-image-grid`.
