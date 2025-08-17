# Protanalyze

Protanalyze is a lightweight bioinformatics tool for **protein physicochemical property analysis**.  
It provides quick, reliable in-silico characterization of protein sequences from FASTA files.

## 🔬 Features
- Molecular weight calculation  
- Isoelectric point (pI) prediction  
- Aromaticity computation  
- Instability index computation  
- GRAVY (hydropathy) score  
- Amino acid composition breakdown (tabular + bar chart)  
- Saves results to a text file and generates plots  

## 🚀 Usage
1. Place your protein sequence in a FASTA file (e.g., `hemo.fasta`).  
2. Run the notebook/script (`analyze.ipynb`).  
3. Outputs:  
   - Console + `hemo_analysis.txt` with detailed properties  
   - `amino_acid_composition.png` plot  

## 📌 Applications
- Protein characterization for drug/vaccine research  
- Comparative proteomics  
- Hypothetical protein analysis  
- Teaching/learning computational biology basics  

## ⚠️ Notes
- Requires Biopython and Matplotlib.  
- Works best with curated FASTA sequences.  
- Extendable for larger datasets and automated pipelines.  

---

**Protanalyze** is designed as a stepping stone for deeper structural, functional, and immunoinformatics studies.
