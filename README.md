# CMVERNet: core code

Core implementation accompanying *CMVERNet* for speech emotion recognition. This release contains the main feature assembly, stable expert, compact expert, context expert, and AURORA routing. It does not contain datasets, audio, pretrained or trained model weights, extracted feature caches, ablation experiments, or experiment results.

## Installation

Use Python 3.10 or newer and install PyTorch for your hardware. From the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The synthetic tests were run with Python 3.13 on CPU. For audio/WavLM extraction, install `python -m pip install -e '.[audio]'` and obtain the external `microsoft/wavlm-base-plus` checkpoint independently. It is not included in this repository; its original revision is pinned in `cmvernet/config.py`. Check the original model and dataset providers' terms before use.

## Inputs and main entry points

All examples use caller-supplied, row-aligned arrays; no datasets or pretrained weights are bundled.

* `cmvernet.features.stable_features(raw, conditioned, egemaps)` assembles the stable input: WavLM raw and conditioned arrays of shape `(N, 13, 1536)`, plus `(N, 88)` eGeMAPS, producing `(N, 40024)`.
* `cmvernet.features.compact_features(static, raw, conditioned)` produces `(N, 53568)`. The `(N, 13632)` static bank is assembled from eight **externally supplied** acoustic components using `cmvernet.acoustic_bank.assemble_acoustic_bank(components, "v9")`. The component feature extractors and original caches are not included.
* `cmvernet.stable_training.run_stable_expert` performs the original five-fold stable-expert training and writes out-of-fold predictions and a summary to a caller-selected output directory. Provide `dataset`, `records`, `features`, integer `labels`, integer `folds` (1–5), a `torch.device`, `results_root`, and explicit `label_order`. The original runner requires each fold to contain exactly 20% of the rows. Each record must have `sample_id`, `relative_path`, and `wav_sha256`; `speaker` and `sentence_id` are used for reference residualization when available. Defaults include 100 epochs and 3,072 selected coordinates.
* `cmvernet.compact.predict_outer_fold(features, labels, speakers, train_indices, test_indices)` fits the compact expert on an outer training fold and predicts the test fold. By default it selects up to 8,192 coordinates and fits a balanced RBF-SVM.
* `cmvernet.context_expert.fit_predict_outer_fold` fits the context branch with inner cross-fitting. It takes row-aligned `base` `(N, 3072)`, `side` `(N, 88)`, integer `labels`, `speakers`, `contents`, `contexts`, temporal `positions`, and outer train/test indices. Prepare the base and side matrices and their normalization with training-fold information; the function does not create them from raw audio.
* `cmvernet.aurora_router` exposes the gate and posterior transport. `cmvernet.historical_routing.route_oof` applies the aggregation used by the historical evaluation runner to aligned out-of-fold expert outputs, fold identifiers, and temporal positions.

For example, train and score one compact fold after preparing your own feature arrays:

```python
from cmvernet.compact import predict_outer_fold
predicted = predict_outer_fold(features, labels, speakers, train_indices, test_indices)
```

The tests use synthetic arrays only and demonstrate the core APIs. Reproducing the reported experimental results additionally requires the original data, dataset-specific fold assignments and metadata, externally supplied feature components and exact software/hardware settings. This repository alone is not an end-to-end reproduction package and no reproduction of the reported scores is claimed.

## Routing implementation note

The paper describes estimating temporal coverage from outer-training metadata. The historical evaluation runner instead computes that coverage from the **evaluation fold's temporal metadata**, without using its labels. `historical_routing.route_oof` retains that historical behavior, including the recorded per-dataset margin settings; it does not establish independent train-only calibration of those margins. `aurora_router.fit_context_gate` can also be called explicitly on training metadata, as described in the paper. The router module's 0.65 default is an original development setting and is distinct from the historical per-dataset values listed in `historical_routing.PER_DATASET_MARGIN`.
