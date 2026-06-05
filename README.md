# GTM

GTM is a graph neural network framework for processing and untangling assembly graphs generated from HiFi data.

## Overview

GTM mainly contains three stages:

1. Construct an assembly graph from HiFi data using hifiasm.
2. Convert the assembly graph into graph data structures for model inference.
3. Apply a trained graph neural network model to infer assembly paths and generate contigs.

## Repository Structure

```text
GTM/
├── checkpoints/                  # Saved training checkpoints
├── configs/                      # Configuration files
├── example/                      # Example input data and outputs
├── Gtm_layers/                   # Neural network layer implementations
├── Gtm_models/                   # Graph neural network model definitions
├── Gtm_utils/                    # Utility functions
├── vendor/                       # External tools installed locally
├── weights/                      # Pretrained model weights
├── create_inference_graphs.py    # Construct graph data for inference
├── generate_data.py              # Generate synthetic training data
├── graph_dataset.py              # Dataset definitions
├── graph_parser.py               # Parse assembly graphs
├── Gtm.py                        # Complete inference pipeline
├── inference.py                  # Inference and contig generation
├── obtain_tools.py               # Install external tools
├── Requirements.yml              # Conda environment file
├── split_data.py                 # Split generated training/validation data
├── train.py                      # Model training script
├── train_valid_chrs.py           # Training/validation chromosome settings
├── LICENSE
└── README.md
```

## Installation

### Requirements

The provided environment has been tested with:

* Python 3.10
* CUDA 11.8
* PyTorch 2.1.2 + CUDA 11.8
* DGL 2.1.0 + CUDA 11.8

The complete reproducible conda environment is provided in:

```text
Requirements.yml
```

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/GTM.git
cd GTM
```

Replace `YOUR_USERNAME` with your GitHub username or the correct repository owner.

### 2. Create the Conda Environment

```bash
conda env create -f Requirements.yml
conda activate GTM
```

### 3. Install External Tools

Install the external tools required by the pipeline:

```bash
python obtain_tools.py
```

## Example

The data needed to run the example consists of simulated *E. coli* reads in gzipped FASTA format. The reads can be found in the `example` directory. The assembly graph will be generated from these reads using hifiasm in GFA format.

Run the following commands from the project root directory.

### 1. Construct the Assembly Graph with hifiasm

This step usually takes less than 1 minute.

```bash
mkdir -p example/hifiasm/output

./vendor/hifiasm-0.18.8/hifiasm \
  --prt-raw \
  -o example/hifiasm/output/ecoli_asm \
  -t 32 \
  -l 0 \
  example/ecoli.fasta.gz
```

### 2. Construct the Necessary Data Structures

This step creates the DGL graphs and auxiliary dictionaries needed for inference. It usually takes less than 1 minute.

```bash
python create_inference_graphs.py \
  --reads example/ecoli.fasta.gz \
  --gfa example/hifiasm/output/ecoli_asm.bp.p_utg.gfa \
  --asm hifiasm \
  --out example
```

The command above will create the following data inside the `example/hifiasm` directory:

* A DGL graph inside `example/hifiasm/processed`
* Auxiliary data inside `example/hifiasm/info`

### 3. Run the Inference Module

This step usually takes less than 1 minute.

```bash
python inference.py \
  --data example \
  --asm hifiasm \
  --out example/hifiasm \
  --model weights/Gtm_weights.pt \
  --device cpu
```

For GPU inference, use `--device cuda:0`.

The edge probabilities will be computed with the default model reported in the paper. The directories `assembly`, `decode`, and `checkpoint` will be created inside `example/hifiasm`.

The final assembly sequence can be found at:

```text
example/hifiasm/assembly/0_assembly.fasta
```

## Usage

To assemble a genome from HiFi data using the complete GTM pipeline, run:

```bash
python Gtm.py \
  -r <reads> \
  -o <out> \
  -t <threads> \
  -m weights/Gtm_weights.pt
```

### Arguments

* `-r, --reads`
  Input reads in FASTA or FASTQ format. Gzip-compressed files are supported.

* `-o, --out`
  Output directory.

* `-t, --threads`
  Number of threads used for running hifiasm.

* `-m, --model`
  Path to the trained model weights.

## Step-by-Step Inference

To run GTM on a new HiFi dataset, first construct an assembly graph in GFA format, then process the graph and perform inference.

### 1. Construct the Assembly Graph from HiFi Reads

GTM uses hifiasm to construct assembly graphs from HiFi reads:

```bash
mkdir -p <out>/hifiasm/output

./vendor/hifiasm-0.18.8/hifiasm \
  --prt-raw \
  -o <out>/hifiasm/output/asm \
  -t <threads> \
  <reads>
```

#### Arguments

* `<reads>`
  Input HiFi data file in FASTA or FASTQ format.

* `<out>`
  Output directory.

* `<threads>`
  Number of threads used for running hifiasm.

### 2. Process the Assembly Graph

Construct DGL graph data and auxiliary information from the reads and the assembly graph:

```bash
python create_inference_graphs.py \
  --reads <reads> \
  --gfa <gfa> \
  --asm hifiasm \
  --out <out>
```

#### Arguments

* `<reads>`
  Input reads in FASTA or FASTQ format.

* `<gfa>`
  Input assembly graph in GFA format.

* `<out>`
  Directory where processed graph data will be saved.

### 3. Generate the Assembly

Run inference using trained model weights:

```bash
python inference.py \
  --data <out> \
  --asm hifiasm \
  --out <out>/hifiasm \
  --model weights/Gtm_weights.pt \
  --device cpu
```

For GPU inference, use `--device cuda:0`.

#### Arguments

* `--data`
  Path to the directory containing the processed graph data.

* `--out`
  Output directory where assembly results will be saved.

* `--model`
  Path to the trained model weights.

* `--device`
  Computing device, for example `cuda:0` or `cpu`.

## Training the Network

### Generate the Training and Validation Data

You can generate synthetic training data by first simulating reads with PBSIM3 and then constructing assembly graphs with hifiasm or Raven. This consists of several steps.

### Step 1. Specify Training and Validation Chromosomes

Specify which chromosomes should be used for the training and validation sets by editing the dictionaries in:

```text
train_valid_chrs.py
```

### Step 2. Prepare Chromosome Reference Files

Since training is performed on individual chromosomes, you also need to save the sequences, or references, of these chromosomes in the following format:

```text
chr1.fasta
chr2.fasta
chr3.fasta
```

The full path to the directory where these chromosome references are stored should be provided to `generate_data.py` using the `--chrdir` argument.

### Step 3. Prepare PBSIM3 Sample Profile Files

PBSIM3 requires sample profile files, such as `sample_profile_ID.fastq` and `sample_profile_ID.stats`, to be stored inside the `vendor/pbsim3` directory.

You can download these files by running:

```bash
bash download_profile.sh
```

The downloaded files correspond to the `sample_profile_ID` specified in `config.py`.

Alternatively, if you already have these files, copy them into `vendor/pbsim3` and edit the value in `config.py` under the key `sample_profile_ID`.

You can also create a new profile by editing the values in `config.py` under the keys `sample_profile_ID` and `sample_file`. Make sure to provide a unique ID for `sample_profile_ID` and a path to an existing FASTQ file for `sample_file`.

For more information, check PBSIM3.

### Step 4. Generate Synthetic Training Data

Finally, run the `generate_data.py` script:

```bash
python generate_data.py \
  --datadir <datadir> \
  --chrdir <chrdir> \
  --asm <asm> \
  --threads <threads>
```

#### Arguments

* `<datadir>`
  Path to the directory where the generated data will be saved.

* `<chrdir>`
  Path to the directory where the chromosome references are stored.

* `<asm>`
  Assembler used for assembly graph construction. Supported options are `hifiasm` and `raven`.

* `<threads>`
  Number of threads used for running the assembler.

### Split the Generated Data into Training and Validation Datasets

Once the data has been generated and stored in the main database, namely the `<datadir>` provided in the previous step, you need to split it into training and validation datasets.

This will copy data from the main database `<datadir>` into `<savedir>`.

Run the following command:

```bash
python split_data.py \
  --datadir <datadir> \
  --savedir <savedir> \
  --name <name> \
  --asm <asm>
```

#### Arguments

* `<datadir>`
  Path to the directory where the generated data is saved.

* `<savedir>`
  Path to the directory where the training and validation datasets will be copied.

* `<name>`
  Name assigned to the training and validation datasets.

* `<asm>`
  Assembler used for assembly graph construction. Supported options are `hifiasm` and `raven`.

Once all the data is copied, the script will print the full paths of the training and validation directories. You can provide those paths as arguments to the `train.py` script.

### Train the Model

```bash
python train.py \
  --train <train> \
  --valid <valid> \
  --asm hifiasm \
  --name <name> \
  --gpu 0
```

#### Arguments

* `<train>`
  Path to the training dataset.

* `<valid>`
  Path to the validation dataset.

* `<name>`
  Name assigned to the trained model.

#### Optional Arguments

* `--overfit`
  Overfit on the training data.

* `--resume`
  Resume training from a checkpoint.

* `--dropout <dropout>`
  Dropout rate used during training.

* `--seed <seed>`
  Random seed used during training.

* `--gpu <gpu>`
  GPU index used for training. If unspecified, training runs on CPU.

By default, training checkpoints are saved in:

```text
checkpoints/
```

## Pretrained Model Weights

Pretrained model weights should be placed in:

```text
weights/
```

The default model path used by GTM is:

```text
weights/Gtm_weights.pt
```

A different trained model can be specified using:

```bash
-m <model>
```

## Reproducibility

All the results in the paper can be reproduced by downloading the relevant data, linked in the manuscript, and following the steps in the Usage section. Use the default model weights available at:

```text
weights/Gtm_weights.pt
```

## License

This project is released under the license described in the `LICENSE` file.

## Citation

If you use GTM in your research, please cite the related work.
