# Structurna

Structurna is an advanced mRNA design and optimization platform that analyzes sequences utilizing both traditional biophysical rules  and state-of-the-art deep learning. The platform enables precise optimization of translation efficiency (TE) while minimizing local biophysical penalties.

---

## Core Features

Structurna utilizes a coordinated **Dual-Engine Architecture** to evaluate and engineer mRNA sequences:

* **Biophysical Penalty Engine**: A rapid, rule-based screening engine that quantifies structural and evolutionary constraints. It evaluates Kozak consensus fidelity, Cap-proximal secondary structures, downstream coding region penalties, and detects unintended decoy start codons or translation-halting motifs. Think of this engine as a strict "surface-level" grader, evaluating a given sequence against a rigid rubric.
* **RiboNN Deep Learning Engine**: An ensemble neural network that evaluates full-length sequence syntax across 5' UTR, CDS, and 3' UTR junctions. Operating at single-nucleotide resolution, it captures non-linear positional dependencies to evaluate precise TE profiles. Think of this model as a Young Sheldon specifically trained to stare at many pictures (an mRNA sequence) and extract the most common and important pixels (biological rules).
* **Optimal Tag Generator**: A heuristic optimization pipeline that scans the local design space to engineer N-terminal nucleotide modifications (multiples of 3 bp) that systematically minimize expression penalties.
* **Joint Structure-Expression Pareto Optimization**: A multi-objective optimization suite that processes engineered variants through multi-fold cross-validation models. It maps the absolute trade-off boundaries between local structural stress and deep-learning predicted translation efficiency, plotting a Pareto-optimal frontier alongside threshold-anchored design quadrants.

---

## Installation

### Prerequisites
* Python 3.10+ (Recommended 3.14+)
* Git

### Setup

1. **Clone the repository and navigate to the root directory:**
   ```bash
   git clone [https://github.com/peewhyyy/Structurna.git](https://github.com/peewhyyy/Structurna.git)
   cd structurna
   ```

2. **Initialize and set up the RiboNN deep learning submodule or dependencies inside the designated subdirectory structure:**
   Ensure your local directory includes the RiboNN library dependencies. The file structure should maintain:
   ```text
   ./RiboNN/src/predict.py
   ./RiboNN/models/
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Deployment

To run the script locally, run the following command:

```bash
python -m streamlit run structurna.py
```

---

## macOS Installation & Setup

Because macOS (particularly Apple Silicon M1/M2/M3/M4 chips) handles Python environments and deep learning dependencies differently than Windows or Linux, follow this optimized setup guide to configure Structurna locally.

### Prerequisites (macOS)
Ensure you have **Homebrew** installed. If you do not have it, open your Terminal and run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Next, install **Git** and **Python 3.14** (recommended for stable deep-learning library support on macOS):
```bash
brew install git python@3.14
```

---

### Step-by-Step Installation

Open your **Terminal** app and execute the following steps:

#### 1. Clone the Repository
```bash
git clone https://github.com/peewhyyy/Structurna.git
cd structurna
```

#### 2. Create and Activate a Virtual Environment
Using a virtual environment is highly recommended on macOS to prevent package conflicts with system-level Python utilities.
```bash
# Create the environment using Python 3.14
python3.14 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```
*(Your terminal prompt will now be prefixed with `(venv)` to indicate the environment is active).*

#### 3. Verify Submodule Directory Structure
Ensure your local directory structure correctly exposes the RiboNN library dependencies:
```text
./RiboNN/src/predict.py
./RiboNN/models/
```

#### 4. Install Dependencies
Install the required packages. On Apple Silicon Macs, pip will automatically resolve and compile packages natively for ARM64:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Apple Silicon Acceleration Note:** If the underlying `RiboNN` deep learning engine utilizes PyTorch, it will automatically attempt to leverage Apple's **MPS (Metal Performance Shaders)** backend to accelerate inference via the Mac's unified memory and GPU, rather than relying on CUDA.

---

## Deployment on macOS

To spin up the platform locally, ensure your virtual environment is active and launch the Streamlit server:

```bash
# Always ensure the virtual environment is active
source venv/bin/activate

# Launch the platform
python -m streamlit run structurna.py
```

This will initialize a local host server and automatically open the Structurna platform interface in your default macOS browser (Safari, Chrome, or Firefox) at `http://localhost:8501`.

---

## Troubleshooting Common macOS Issues

* **`xcrun: error: invalid active developer path`**
  If you see this error after a macOS system update, your Xcode command-line tools have become detached. Reinstall them by running:
  ```bash
  xcode-select --install
  ```
* **Architecture Conflicts (`mach-o file, but is an incompatible architecture`)**
  If a Python package throws an architecture error, ensure you are running a native ARM64 version of Python, rather than an Intel version running via Rosetta emulation. You can verify your native state by running `arch` in your terminal; it should output `arm64`.

---

## Advanced Usage & Optimization Workflows

### 1. Calibration and Baseline Evaluation
Input your baseline 5' UTR, CDS, and 3' UTR configurations. Fine-tune your biophysical scoring coefficients (Minimum Free Energy offsets, Kozak multipliers, and Cap parameters) to evaluate structural baseline stress.

### 2. Multi-Objective Optimization Space
* Select your target tag length constraints and background species model (human or mouse).
* Isolate regional or target evaluation organs to filter cross-validation predictions against tissue-specific background profiles.
* Run the validation pipeline to project variants across the **Reference Threshold Quadrants**:
  * **Strict Win**: Identifies engineered candidates achieving lower biophysical penalties and higher predicted TE than the original sequence.
  * **Expression Upgrade**: Pinpoints candidates maximizing translation yield where structural adjustments are permissible.
