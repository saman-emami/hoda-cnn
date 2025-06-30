# Handwritten Persian Digits Recognition using CNN

A comprehensive TensorFlow/Keras implementation for training a Convolutional Neural Network to recognize handwritten Persian digits (۰-۹) using the Hoda dataset. 
The model achieves an accuracy of 99.66% over the test data.

## Requirements

The project requires the following Python libraries. You can install them using a `requirements.txt` file.

* `opencv-python`
* `tensorflow`
* `numpy`
* `matplotlib`
* `scikit-learn`

## Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saman-emami/hoda-cnn
    cd hoda-cnn
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **Run the script:**
    Execute the Jupyter Notebook to train the model and see the results.
---

## Command-Line Interface
The repository includes a CLI tool (hoda2npz) that converts Hoda `.cdb` files into machine learning-ready NumPy archives.

### Basic conversion:
```bash
python -m hoda2npz.cli file1.cdb file2.cdb -o dataset.npz
```

### With custom settings:
```bash
python -m hoda2npz.cli *.cdb -o hoda_32x32.npz --size 32x32 --normalize -v
```

### Arguments:

| Argument | Description | Default |
| :-- | :-- | :-- |
| `cdb_files` | Input .cdb files to process | Required |
| `-o, --output` | Output .npz file path | Required |
| `--size` | Target dimensions (WxH format) | 28x28 |
| `--normalize` | Scale pixels to 0-1 range | False |
| `-v, --verbose` | Show detailed progress | False |

### Output Format:

Creates a compressed .npz file containing:

* `images`: Float32 array (N, H, W, 1) with processed images
* `labels`: Int16 array (N,) with digit labels


## Acknowledgments
The logic for the self-contained `.cdb` data reader, found in the `cdb_processor.py` file, is inspired by the [**HodaDatasetReader**](https://github.com/amir-saniyan/HodaDatasetReader) project on GitHub.
