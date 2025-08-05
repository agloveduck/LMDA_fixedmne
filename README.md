````markdown
# LMDA-Net for BCI Competition IV-2a Dataset

> This repository is a modified and enhanced implementation based on the original [LMDA-Code](https://github.com/MiaoZhengQing/LMDA-Code).

## 🧠 Overview

This project applies the LMDA (Low-Rank Multi-Domain Adaptation) model to the BCI Competition IV-2a motor imagery dataset. It includes preprocessing (bandpass filtering, normalization, Euclidean alignment), model training, and evaluation with customized metrics.

## Dataset bci2a(.gdf&.mat)
https://pan.baidu.com/s/16K8WGUC-_KOq27rNzmF1qA 提取码: 6i5h
## 📁 Code Structure

- `cal_result.py`: Calculates average accuracy ± std and Cohen's Kappa for each subject. Computes metrics based on `Best test acc` and `Mean acc of last 10 epochs`.
- `compared_models.py`: Contains:
  - `weights_init`: Weight initialization utility.
  - `MaxNormDefaultConstraint`: Max-norm regularization class.
  - Two baseline models.
- `data_loader.py`: Modified `load`, `extract_data`, `extract_events`, `extract_segment_trial` functions for compatibility with high versions of `mne`.
- `data_preprocess.py`: Implements preprocessing steps adapted from the [SDDA paper](https://arxiv.org/pdf/2202.09559.pdf), including:
  - Bandpass filtering
  - Normalization
  - Euclidean Alignment (EA)
- `experiment.py`: Contains training and evaluation functions.
  - `Measurement` class includes:
    - `max_mean_offset`
    - `index_equation` to fix runtime issues.
- `lmda_model.py`: The LMDA model definition.
- `main_BCIIV2a.py`: Script to train the model on the 2a dataset. Run with:
  ```bash
  python main_BCIIV2a.py
````

Adjusted to align with the updated data loading pipeline.

* `model_paper.py`: Alternate LMDA model implementation.

## 🔧 Dependencies

* `mne==1.10.0`
* Tested on: **RTX 3090 (24GB GPU)**

## 📊 Experimental Results

### Example Logs (Excerpt)

```text
2025/08/04 17:42:52 INFO : Best test acc 85.06944 
2025/08/04 17:42:52 INFO : mean acc of last 10 epochs 82.04861 
2025/08/04 17:42:52 INFO : std of last 10 epochs 0.90611 
2025/08/04 17:42:52 INFO : offset: max SUB mean acc 3.02083 
2025/08/04 17:42:52 INFO : Index score(0.4*max+0.4*mean-0.2*offset): 66.24306
```

### 📈 Performance on BCI Competition IV-2a Dataset

#### Based on **Best Test Accuracy**:

| Methods             | A01  | A02  | A03  | A04  | A05  | A06  | A07  | A08  | A09  | Average acc ± std (Kappa)     |
| ------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------------------------- |
| **LMDA-Net (ours)** | 85.1 | 64.9 | 89.9 | 78.1 | 65.6 | 61.8 | 92.7 | 82.6 | 85.4 | **78.5 ± 11.6 (0.71 ± 0.15)** |

#### Based on **Mean Accuracy of Last 10 Epochs**:

| Methods             | A01  | A02  | A03  | A04  | A05  | A06  | A07  | A08  | A09  | Average acc ± std (Kappa)     |
| ------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------------------------- |
| **LMDA-Net (ours)** | 82.1 | 62.5 | 88.2 | 76.3 | 61.3 | 56.3 | 90.0 | 79.1 | 83.7 | **75.5 ± 12.4 (0.67 ± 0.17)** |



## 📎 Reference

This codebase is based on the official [LMDA repository](https://github.com/MiaoZhengQing/LMDA-Code). All credit to the original authors.

