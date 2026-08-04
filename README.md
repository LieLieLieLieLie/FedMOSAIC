# FedMOSAIC

Official implementation of **FedMOSAIC: Memory-Efficient Orthogonal Simplex
Anchors with Integrated Consolidation for Task-Free Asynchronous Federated
Continual Intrusion Detection**.

FedMOSAIC addresses federated continual intrusion detection when attack classes
arrive in client-dependent orders, local feature distributions drift, only a
subset of clients participates, and updates may have different model ages. The
method combines regular-simplex anchors, classwise sufficient statistics,
stable-plastic projection, reliability-staleness-aware aggregation, replay-free
server consolidation, and statistics-conditioned calibration.

This repository contains the complete Python pipeline for deterministic data
preprocessing, federated simulation, baseline evaluation, statistical
summarization, prediction extraction, and paper-style PDF visualization. Dataset
archives, processed data, trained checkpoints, and generated results are not
included.

## Requirements

- Python 3.11 is recommended.
- PyTorch 2.0 or newer.
- A CUDA-capable device is optional. The implementation selects CUDA when it is
  available and otherwise falls back to CPU.

Create an isolated environment and install the dependencies:

```bash
git clone https://github.com/LieLieLieLieLie/FedMOSAIC.git
cd FedMOSAIC
python -m venv .venv
```

Activate the environment, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset acquisition

The experiments use labeled, tabular intrusion-detection data derived from two
public benchmarks:

- [Edge-IIoTset](https://ieee-dataport.org/8939), originally released for
  centralized and federated IoT/IIoT intrusion detection.
- [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html), collected
  from a 105-device IoT topology under benign traffic and 33 attacks.

For exact reproduction, download the following six-class federated archives
from the public [Zenodo record](https://doi.org/10.5281/zenodo.11315294):

- `df_FL_Edge-IIoTset_6_classes.rar`
- `df_FL_CICIoT2023_6_classes.rar`

Extract the CSV files and create this directory structure at the repository
root:

```text
FedMOSAIC/
|-- data/
|   |-- EdgeIIoTset/
|   |   `-- df_FL_Edge-IIoTset_6_classes.csv
|   `-- CICIoT2023/
|       `-- df_FL_CICIoT2023_6_classes.csv
|-- config.py
|-- prepare_data.py
`-- ...
```

The spelling and capitalization of both directory and file names must match the
tree above. The code does not redistribute either dataset; users remain
responsible for complying with the providers' access and usage conditions.

## Preprocessing

Prepare both datasets with the protocol used in the experiments:

```bash
python prepare_data.py --all
```

The command performs deterministic classwise sampling, a stratified
train/validation/test split, training-only robust scaling over the 5th--95th
percentile range, and clipping to `[-12, 12]`. Processed arrays and preprocessing
metadata are written to `data/processed/`.

To change the per-class sample count or preprocessing seed:

```bash
python prepare_data.py --all --per-class 6000 --seed 2026
```

## Quick verification

Run a short FedMOSAIC experiment before launching the complete benchmark:

```bash
python run_experiment.py \
  --dataset edgeiiot \
  --method fedmosaic \
  --seed 0 \
  --rounds 2 \
  --local-steps 1 \
  --overwrite
```

Available dataset identifiers are `edgeiiot` and `ciciot`. Available method
identifiers are `fedavg_er`, `glfc`, `evofedids`, `fedta`, `fedagc`, and
`fedmosaic`.

## Reproducing the evaluation

Run the experiment suites separately so that completed configurations can be
cached and resumed:

```bash
python run_all.py --suite main
python run_all.py --suite stress
python run_all.py --suite ablation
python run_all.py --suite sensitivity
```

Use `python run_all.py --suite all` to execute every suite. By default, an
existing completed checkpoint is reused; pass `--overwrite` only when a run
must be recomputed.

After training, generate the numerical summaries, cached predictions, and PDF
figures:

```bash
python summarize_results.py
python extract_predictions.py
python visualize_results.py
```

## Output structure

The scripts create all output directories automatically:

```text
results/
|-- figures/   # vector PDF figures
|-- models/    # checkpoints, histories, metrics, and prediction caches
|-- tables/    # CSV and LaTeX-ready numerical summaries
`-- logs/      # execution logs
```

Visualization reads cached artifacts and does not retrain the models. The five
baselines and FedMOSAIC share the same tabular encoder, evolving stream, client
sampling process, optimizer budget, and evaluation implementation. Replay and
other method-specific auxiliary states are included in the reported memory and
communication accounting.

## Reproducibility notes

- Main comparisons use seeds 0, 1, and 2.
- Stress tests vary client participation, Dirichlet label skew, and maximum
  staleness.
- The code records the complete experiment configuration in every JSON history.
- Random generators and deterministic backend options are initialized from the
  experiment seed.
- A CPU run is supported but will generally take longer than CUDA execution.

## Citation

The accompanying manuscript is under peer review. If this implementation is
useful in your research, please cite the paper by its title; complete
bibliographic information will be added after publication.
