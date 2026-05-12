# Pharmaceutical AI Portfolio

A production-ready portfolio demonstrating machine learning and computational chemistry for drug discovery. Built by a software engineer transitioning into pharmaceutical AI/ML roles.

**Live Portfolio:** [View Website](https://portfolio-website-ecru-pi-95.vercel.app)

## What This Portfolio Demonstrates

This repository contains working Python modules that solve real problems in drug discovery:

1. **ML-Based Toxicity Prediction** - Random Forest classifier predicting molecular toxicity
2. **Fragment-Based Drug Design** - BRICS/RECAP decomposition and lead optimization
3. **Molecular Property Analysis** - Lipinski's Rule validation and druglikeness scoring
4. **Molecular Docking Preparation** - 3D structure generation and conformer analysis

Each module is production-ready with demos, documentation, and outputs.

## Quick Start

```bash
git clone https://github.com/Zulekha/pharma-ai-portfolio.git
cd pharma-ai-portfolio
conda env create -f environment.yml
conda activate pharma-ai-env

# Run demos
python src/ml_compound_prioritisation.py
python src/fragment_based_drug_design.py
python src/molecular_property_analyzer.py
python src/molecular_docking_prep.py
```

## Core Modules

### 1. ML Compound Prioritization

**File:** `src/ml_compound_prioritisation.py`

Trains a Random Forest classifier to predict molecular toxicity using Morgan fingerprints.

**Key Features:**
- Morgan fingerprint generation (2048-bit)
- Binary classification (toxic/safe)
- Model evaluation (Accuracy, ROC-AUC, Precision, Recall)
- Compound ranking by safety score
- Model persistence (pickle)

**Output Example:**
```
Performance Metrics:
  Accuracy:  0.500
  Precision: 0.500
  Recall:    1.000
  ROC-AUC:   1.000

Compound Ranking (by Safety):
Rank  1 | SAFE  | Safety: 0.900 | Ibuprofen
Rank  2 | SAFE  | Safety: 0.800 | Caffeine
Rank  3 | TOXIC | Safety: 0.120 | Benzoquinone
```

**Business Impact:** Prioritize safer compounds early, reducing late-stage failures.

---

### 2. Fragment-Based Drug Design

**File:** `src/fragment_based_drug_design.py`

Decomposes molecules into fragments using BRICS/RECAP methods for lead optimization.

**Key Features:**
- BRICS and RECAP fragmentation
- Murcko scaffold extraction
- Fragment library generation with filtering
- Lead compound analysis
- Lipinski violation detection
- Optimization suggestions

**Output Example:**
```
Fragment Library: 10 fragments generated
Valid drug-like: 9/10

Lead Analysis - Ibuprofen:
  MW: 206.28, LogP: 3.07
  Lipinski Violations: 0
  
Optimization Suggestions:
  1. Add polar groups to increase TPSA
  2. Lead compound already drug-like
```

**Business Impact:** Systematically explore chemical space around lead compounds.

---

### 3. Molecular Property Analyzer

**File:** `src/molecular_property_analyzer.py`

Calculates pharmaceutical properties and evaluates drug-likeness.

**Key Features:**
- Lipinski's Rule of Five validation
- QED (Quantitative Estimate of Druglikeness) scoring
- Batch processing with CSV export
- 10+ molecular descriptors (MW, LogP, TPSA, HBD/HBA)

**Properties Calculated:**
- Molecular Weight
- LogP (lipophilicity)
- TPSA (polar surface area)
- H-bond donors/acceptors
- Rotatable bonds
- Aromatic/aliphatic rings
- QED score

**Output:** CSV files ready for downstream ML pipelines.

---

### 4. Molecular Docking Preparation

**File:** `src/molecular_docking_prep.py`

Prepares ligands for molecular docking studies.

**Key Features:**
- 3D structure generation
- Multiple conformer generation (with energy minimization)
- Partial charge assignment (Gasteiger)
- File export (PDB, SDF formats)
- Energy calculations (MMFF94, UFF)

**Output Example:**
```
Aspirin Preparation:
  3D structure generated: 21 atoms
  Conformers: 2 generated
  Energy range: 20.83 to 20.83 kcal/mol
  Files: aspirin.pdb, aspirin.sdf
```

**Business Impact:** Streamline ligand preparation for virtual screening campaigns.

---

## Project Structure

```
pharma-ai-portfolio/
├── src/                              # Core Python modules
│   ├── ml_compound_prioritisation.py
│   ├── fragment_based_drug_design.py
│   ├── molecular_property_analyzer.py
│   └── molecular_docking_prep.py
├── output/                           # Generated outputs
│   ├── toxicity_model.pkl
│   ├── fragment_library.csv
│   └── molecular_analysis_results.csv
├── notebooks/                        # Jupyter demos (coming soon)
├── alphafold_target_pipeline/        # AlphaFold + docking workflow
├── bioactivity-prediction/           # ChEMBL ML pipeline
├── requirements.txt
├── environment.yml
└── README.md
```

## Technical Stack

**Core Libraries:**
- **RDKit** - Cheminformatics and molecular manipulation
- **Scikit-learn** - Machine learning (Random Forest)
- **Pandas/NumPy** - Data processing
- **Matplotlib** - Visualization (coming)

**Techniques:**
- Molecular fingerprints (Morgan, MACCS)
- BRICS/RECAP fragmentation
- MMFF94/UFF force fields
- Lipinski's Rule of Five
- QED scoring

## Real-World Applications

### Drug Discovery Pipeline Integration

1. **Early Hit Identification**
   - Screen virtual libraries with toxicity predictor
   - Filter by Lipinski's Rule
   - Prioritize drug-like compounds

2. **Lead Optimization**
   - Fragment-based design for SAR exploration
   - Property optimization (reduce MW, LogP)
   - Generate focused libraries

3. **Virtual Screening**
   - Prepare ligands for docking
   - Generate multiple conformers
   - Export to AutoDock Vina format

## Why DHFR?

Dihydrofolate Reductase (DHFR) is used as the biological context throughout this portfolio because:
- Clinically validated target (cancer, infectious disease)
- Well-characterized structure and mechanism
- Established therapeutics (methotrexate, trimethoprim)
- Ideal for demonstrating structure-based drug discovery

## AlphaFold Integration

The `alphafold_target_pipeline/` demonstrates:
- Protein structure prediction (ColabFold)
- Binding site identification
- Ligand docking with AutoDock Vina
- Pose analysis and visualization

**Target:** DHFR enzyme structure prediction and known inhibitor docking validation.

## ML Bioactivity Prediction

The `bioactivity-prediction/` project shows:
- ChEMBL data retrieval and processing
- Molecular descriptor generation
- Random Forest binary classification
- ROC-AUC evaluation
- Production-ready prediction pipeline

**Result:** 85-90% accuracy predicting DHFR inhibition from molecular structure.

## Coming Soon

- [ ] Jupyter notebooks with visualization
- [ ] ADMET property prediction
- [ ] Binding pose clustering analysis
- [ ] Extended toxicity datasets (Tox21)
- [ ] Deep learning for activity prediction

## Installation

**Requirements:**
- Python 3.8+
- Conda (recommended)

**Setup:**
```bash
# Clone repository
git clone https://github.com/Zulekha/pharma-ai-portfolio.git
cd pharma-ai-portfolio

# Create environment
conda env create -f environment.yml
conda activate pharma-ai-env

# Test installation
python src/molecular_property_analyzer.py
```

## Usage Examples

**Analyze a molecule:**
```python
from src.molecular_property_analyzer import MolecularPropertyAnalyzer

analyzer = MolecularPropertyAnalyzer()
results = analyzer.analyze_molecule('CC(=O)Oc1ccccc1C(=O)O')  # Aspirin
print(f"QED Score: {results['QED']:.3f}")
print(f"Passes Lipinski: {results['Lipinski']['LipinskiPass']}")
```

**Predict toxicity:**
```python
from src.ml_compound_prioritisation import ToxicityPredictor

predictor = ToxicityPredictor.load_model('output/toxicity_model.pkl')
results = predictor.predict(['CC(C)Cc1ccc(cc1)C(C)C(=O)O'])  # Ibuprofen
print(results[['SMILES', 'Safety_Score', 'Predicted_Toxic']])
```

**Generate fragments:**
```python
from src.fragment_based_drug_design import FragmentLibraryGenerator

generator = FragmentLibraryGenerator()
fragments = generator.decompose_brics('CC(=O)Oc1ccccc1C(=O)O')
print(f"Generated {len(fragments)} fragments")
```

## About

**Author:** Zuleikha Khan  
**Background:** Software Engineer (7+ years) transitioning to AI/ML in pharmaceutical R&D  
**Education:** Postgraduate Diploma in AI (Distinction), National College of Ireland  
**Focus:** Drug discovery, computational chemistry, machine learning for life sciences

**Portfolio Website:** https://portfolio-website-ecru-pi-95.vercel.app  
**LinkedIn:** [Connect](https://www.linkedin.com/in/zuleikha-khan)  
**Location:** Dublin, Ireland

## License

MIT License - See LICENSE file for details

---

**For Recruiters:** This portfolio demonstrates production-ready code for pharmaceutical AI roles. All modules are tested, documented, and produce real outputs. The work shows practical understanding of drug discovery workflows, not just theoretical ML knowledge.