# Structurna

Structurna is an advanced mRNA design and optimization platform that bridges traditional biophysical rules with state-of-the-art deep learning syntax analysis. Built to accelerate translation engineering, the platform enables precise optimization of translation efficiency (TE) while minimizing local biophysical penalties.

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

## Advanced Usage & Optimization Workflows

### 1. Calibration and Baseline Evaluation
Input your baseline 5' UTR, CDS, and 3' UTR configurations. Fine-tune your biophysical scoring coefficients (Minimum Free Energy offsets, Kozak multipliers, and Cap parameters) to evaluate structural baseline stress.

### 2. Multi-Objective Optimization Space
* Select your target tag length constraints and background species model (human or mouse).
* Isolate regional or target evaluation organs to filter cross-validation predictions against tissue-specific background profiles.
* Run the validation pipeline to project variants across the **Reference Threshold Quadrants**:
  * **Strict Win**: Identifies engineered candidates achieving lower biophysical penalties and higher predicted TE than the original sequence.
  * **Expression Upgrade**: Pinpoints candidates maximizing translation yield where structural adjustments are permissible.
