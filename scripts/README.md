# Usage Examples

## Generating Synthetic Time Series (KernelSynth)

- Install this package with with the `training` extra:
    ```
    pip install "chronos-forecasting[training] @ git+https://github.com/amazon-science/chronos-forecasting.git"
    ```
- Run `kernel-synth.py`:
    ```sh
    # With defaults used in the paper (1M time series and 5 max_kernels)
    python kernel-synth.py

    # You may optionally specify num-series and max-kernels
    python kernel-synth.py \
        --num-series <num of series to generate> \
        --max-kernels <max number of kernels to use per series>
    ```
    The generated time series will be saved in a [GluonTS](https://github.com/awslabs/gluonts)-comptabile arrow file `kernelsynth-data.arrow`.

## Pretraining (and fine-tuning) Chronos models
- Install this package with with the `training` extra:
    ```
    pip install "chronos-forecasting[training] @ git+https://github.com/amazon-science/chronos-forecasting.git"
    ```
- Convert your time series dataset into a GluonTS-compatible file dataset. We recommend using the arrow format. You may use the `convert_to_arrow` function from the following snippet for that. Optionally, you may use [synthetic data from KernelSynth](#generating-synthetic-time-series-kernelsynth) to follow along.
    ```py
    from pathlib import Path
    from typing import List, Union

    import numpy as np
    from gluonts.dataset.arrow import ArrowWriter


    def convert_to_arrow(
        path: Union[str, Path],
        time_series: Union[List[np.ndarray], np.ndarray],
        compression: str = "lz4",
    ):
        """
        Store a given set of series into Arrow format at the specified path.

        Input data can be either a list of 1D numpy arrays, or a single 2D
        numpy array of shape (num_series, time_length).
        """
        assert isinstance(time_series, list) or (
            isinstance(time_series, np.ndarray) and
            time_series.ndim == 2
        )

        # Set an arbitrary start time
        start = np.datetime64("2000-01-01 00:00", "s")

        dataset = [
            {"start": start, "target": ts} for ts in time_series
        ]

        ArrowWriter(compression=compression).write_to_file(
            dataset,
            path=path,
        )


    if __name__ == "__main__":
        # Generate 20 random time series of length 1024
        time_series = [np.random.randn(1024) for i in range(20)]

        # Convert to GluonTS arrow format
        convert_to_arrow("./noise-data.arrow", time_series=time_series)
    ```
- Modify the [training configs](training/configs) to use your data. Let's use the KernelSynth data as an example.
    ```yaml
    # List of training data files
    training_data_paths:
    - "/path/to/kernelsynth-data.arrow"
    # Mixing probability of each dataset file
    probability:
    - 1.0
    ```
    You may optionally change other parameters of the config file, as required. For instance, if you're interested in fine-tuning the model from a pretrained Chronos checkpoint, you should change the `model_id`, set `random_init: false`, and (optionally) change other parameters such as `max_steps` and `learning_rate`.
- Start the training (or fine-tuning) job:
    ```sh
    # On single GPU
    CUDA_VISIBLE_DEVICES=0 python training/train.py --config /path/to/modified/config.yaml

    # On multiple GPUs (example with 8 GPUs)
    torchrun --nproc-per-node=8 training/train.py --config /path/to/modified/config.yaml

    # Fine-tune `amazon/chronos-t5-small` for 1000 steps with initial learning rate of 1e-3
    CUDA_VISIBLE_DEVICES=0 python training/train.py --config /path/to/modified/config.yaml \
        --model-id amazon/chronos-t5-small \
        --no-random-init \
        --max-steps 1000 \
        --learning-rate 0.001
    ```
    The output and checkpoints will be saved in `output/run-{id}/`.
> [!TIP]  
> If the initial training step is too slow, you might want to change the `shuffle_buffer_length` and/or set `torch_compile` to `false`.

> [!IMPORTANT]  
> When pretraining causal models (such as GPT2), the training script does [`LastValueImputation`](https://github.com/awslabs/gluonts/blob/f0f2266d520cb980f4c1ce18c28b003ad5cd2599/src/gluonts/transform/feature.py#L103) for missing values by default. If you pretrain causal models, please ensure that missing values are imputed similarly before passing the context tensor to `ChronosPipeline.predict()` for accurate results.
- (Optional) Once trained, you can easily push your fine-tuned model to HuggingFace🤗 Hub. Before that, do not forget to [create an access token](https://huggingface.co/settings/tokens) with **write permissions** and put it in `~/.cache/huggingface/token`. Here's a snippet that will push a fine-tuned model to HuggingFace🤗 Hub at `<your_hf_username>/chronos-t5-small-fine-tuned`.
    ```py
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained("/path/to/fine-tuned/model/ckpt/dir/")
    pipeline.model.model.push_to_hub("chronos-t5-small-fine-tuned")
    ```

## Evaluating Chronos models

### Standard Evaluation (Paper Benchmarks)

Follow these steps to compute the WQL and MASE values for the in-domain and zero-shot benchmarks in our paper.

- Install this package with with the `evaluation` extra:
    ```
    pip install "chronos-forecasting[evaluation] @ git+https://github.com/amazon-science/chronos-forecasting.git"
    ```
- Run the evaluation script:
    ```sh
    # In-domain evaluation
    # Results will be saved in: evaluation/results/chronos-t5-small-in-domain.csv
    python evaluation/evaluate.py evaluation/configs/in-domain.yaml evaluation/results/chronos-t5-small-in-domain.csv \
        --chronos-model-id "amazon/chronos-t5-small" \
        --batch-size=32 \
        --device=cuda:0 \
        --num-samples 20

    # Zero-shot evaluation
    # Results will be saved in: evaluation/results/chronos-t5-small-zero-shot.csv
    python evaluation/evaluate.py evaluation/configs/zero-shot.yaml evaluation/results/chronos-t5-small-zero-shot.csv \
        --chronos-model-id "amazon/chronos-t5-small" \
        --batch-size=32 \
        --device=cuda:0 \
        --num-samples 20
    ```
- Use the following snippet to compute the aggregated relative WQL and MASE scores:
    ```py
    import pandas as pd
    from scipy.stats import gmean  # requires: pip install scipy


    def agg_relative_score(model_df: pd.DataFrame, baseline_df: pd.DataFrame):
        relative_score = model_df.drop("model", axis="columns") / baseline_df.drop(
            "model", axis="columns"
        )
        return relative_score.agg(gmean)


    result_df = pd.read_csv("evaluation/results/chronos-t5-small-in-domain.csv").set_index("dataset")
    baseline_df = pd.read_csv("evaluation/results/seasonal-naive-in-domain.csv").set_index("dataset")

    agg_score_df = agg_relative_score(result_df, baseline_df)
    ```

### Local Evaluation with Predictions

The `evaluate_local_with_predictions.py` script provides a comprehensive local evaluation workflow that compares fine-tuned models against zero-shot baselines using both steps-ahead and rolling origin approaches.

#### Features

- **Fine-tuned vs Zero-shot Comparison**: Evaluates your trained model against the base Chronos model
- **Multiple Prediction Strategies**: 
  - **Steps Ahead (SA)**: Predicts next N values from the end of the time series
  - **Rolling Origin (RO)**: Makes multiple predictions from different starting points and averages them
- **Comprehensive Output**: Saves results with clear labels for training data, validation data, and all prediction types
- **Inference Configuration**: Automatically loads saved inference parameters from training runs

#### Usage Examples

**Basic Usage (Climate Dataset Example):**
```sh
# Evaluate a fine-tuned climate model
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5
```

**Custom Prediction Length:**
```sh
# Predict next 10 values instead of 5
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions_10steps.csv" \
    --prediction-length 10
```

**Using Trained Model Directly (instead of inference model):**
```sh
# Use the actual trained model instead of a larger inference model
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5 \
    --no-use-inference-model
```

**Disable Zero-shot Comparison:**
```sh
# Only evaluate fine-tuned model (faster execution)
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5 \
    --no-enable-zero-shot
```

#### Output Format

The script generates a CSV file with the following columns:
- `timestamp`: Time index for each value
- `value`: Numerical value (actual or predicted)
- `type`: Label indicating the data type:
  - `actual`: Training and validation data
  - `predicted FT SA`: Fine-tuned Steps Ahead predictions
  - `predicted ZR SA`: Zero-shot Steps Ahead predictions  
  - `predicted FT RO`: Fine-tuned Rolling Origin predictions
  - `predicted ZR RO`: Zero-shot Rolling Origin predictions

#### Understanding the Results

1. **Training Data**: Historical values used for model training
2. **Validation Data**: Last N values (where N = prediction_length) - these are the "ground truth" for comparison
3. **Predictions**: Four types of forecasts for the validation period:
   - **FT SA**: Your fine-tuned model's direct prediction
   - **ZR SA**: Base Chronos model's direct prediction
   - **FT RO**: Your fine-tuned model's rolling origin prediction (more robust)
   - **ZR RO**: Base Chronos model's rolling origin prediction

#### Performance Tips

- **GPU Usage**: The script automatically uses GPU if available
- **Memory**: For large models, consider using `--no-use-inference-model` to use your trained model directly
- **Speed**: Disable zero-shot evaluation with `--no-enable-zero-shot` for faster execution
- **Rolling Origin**: The script uses 3 rolls by default for rolling origin predictions, which provides more robust forecasts

#### Troubleshooting

- **Model Loading Issues**: Ensure your model path points to a valid checkpoint directory
- **Dataset Format**: Make sure your dataset is in GluonTS Arrow format
- **Memory Errors**: Try using a smaller model or reduce prediction length
- **Inference Config**: The script automatically looks for saved inference parameters in `results/{dataset_name}/{column_name}/`