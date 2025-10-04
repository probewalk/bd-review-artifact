# Anonymized Artifact (Review-Only)

This repository is an **anonymized, review-only** artifact for the WWW research track.  
We have removed any information that could reveal author identity to preserve **double-blind fairness**.  
Please do **not** redistribute this URL outside the PC/AC.

---

## 1. What’s included / not included

### Included
- **Core algorithm implementation** (our method).
- **Complete experiment harness** (single entry script to run experiments).
- **Graph preprocessing utilities** (cleaning and converting raw graphs into the internal format).
- **Full reproducibility guide** (step-by-step instructions to replicate the experiments).

### Not included (and why)
- **Third-party datasets**: we do **not** have the right to redistribute external data; please download from the original sources.
- **Closed-source baselines**: the authors of **PUSH**, **STW**, and **SWF** did not publicly release their code. We obtained implementations **privately for academic comparison only**, therefore we cannot redistribute them.  
  To keep our runner compatible, we provide **empty stubs**: `push.py`, `stw.py`, `swf.py` (no content).  
  For the STW/SWF baseline description, see:  
  *Changan Liu, Ahad N. Zehmakan, and Zhongzhi Zhang. 2024. Fast Query of Biharmonic Distance in Networks. In Proceedings of KDD’24, 1887–1897.*

---

## 2. Reproduction guide

> **Project root** below refers to the top-level directory of this repository.  
> Paths in all commands are **relative to the project root**.

### 2.1 Quick start: toy example (no external data)

**(1) Create and activate a clean environment**

```bash
# Create a conda env from requirements.txt (example: Python 3.10)
conda create -n ml-gpu python=3.10 -y
conda activate ml-gpu

# Install Python deps
pip install -r requirements.txt

#Change into the project root (if not already there):
cd <path-to-project-root>

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

**(2) Run the toy experiment**
A small toy graph is provided, so you can run end-to-end without downloading datasets:
```bash
python src/experiment/run_experiment.py
```
Outputs will appear under:data/paper/test/toy/; File naming follows: <method>_<relative_error>_<elapsed_ms>.json. For example, probewalk_0.017385_1928.3.json means method = probewalk, relative error = 0.017385 (≈1.74%), elapsed ≈ 1928.3 ms.


### 2.2 Experiment 1 — Query Efficiency on Real-World Networks
Goal. Reproduce the six-graph comparison.

**Step 1 — Download datasets (from official sources)**

- Facebook: https://snap.stanford.edu/data/ego-Facebook.html
- DBLP: https://snap.stanford.edu/data/com-DBLP.html
- YouTube: https://snap.stanford.edu/data/com-Youtube.html
- AS-Skitter: https://snap.stanford.edu/data/as-Skitter.html
- Orkut: https://snap.stanford.edu/data/com-Orkut.html
- LiveJournal: https://snap.stanford.edu/data/com-LiveJournal.html

Place raw files under: data/real-data/<dataset>/.



We use a simple internal text format for graphs:
- First line: number of nodes n
- Subsequent lines: one edge per line, u v (space-separated), 0/1-based is acceptable as long as consistent.


If you already have .txt in <u v> per line, clean & normalize: python tool/prepare_real_data/txt2txt.py   (The names of the graphs need to be modified manually)
If the source is CSV/MTX, use the corresponding converters:  python tool/prepare_real_data/csv2txt.py   or python tool/prepare_real_data/mtx2txt.py 



**Step 2 — Build graph metadata**

We store metadata in data/real-data/info.json. Use:

```bash
# Basic graph statistics (node/edge counts, etc.)
python tool/get_info/get_graph_info.py
# (Optional) Spectral quantity λ, if needed for your configuration
python tool/get_info/get_lambda.py
# Sample 100 query pairs and write them into info.json
python tool/get_info/generate_random_pair.py
```


**Step 3 — (Optional) Produce ground truth**

For high-precision references we used a gold executor that calls a strong baseline (PUSH).
Because we cannot redistribute that implementation, you must supply your own if you wish to regenerate ground truth.


To switch to the gold executor:
```python
# In src/experiment/run_experiment.py:
#   comment out:     executor = ExperimentExecutor(methods_registry)
#   uncomment:       executor = ExperimentExecutor_gold(methods_registry)

```
Then run:
```bash
python src/experiment/run_experiment.py
```
Ground-truth values will be written back into data/real-data/info.json.

**Step 4 — Run the efficiency comparison**
Switch back to the standard executor:
```python
# In src/experiment/run_experiment.py:
#   uncomment:       executor = ExperimentExecutor(methods_registry)
#   comment out:     executor = ExperimentExecutor_gold(methods_registry)

```
Prepare a config (see src/config/config.example.json) and run:
```bash
python src/experiment/run_experiment.py
```


### 2.3 Experiment 2 — Query Scalability on Large-Scale Networks
Goal. Run the large-graph study with doubling over truncation length L.


**Step 1 — Download datasets**

- Bluesky (directed; treat as undirected in our runs): dataset available via Social Media Archive (access may require approval).  https://socialmediaarchive.org/record/78?utm_source=chatgpt.com&v=pdf
- Twitter-2010 (directed): https://snap.stanford.edu/data/twitter-2010.html
- Friendster: https://snap.stanford.edu/data/com-Friendster.html

Place raw files under: data/real-data/<dataset>/  ;  Preprocess to .txt as in §2.2.


**Step 2 — Build metadata and sample queries**

```bash
python tool/get_info/get_graph_info.py
python tool/get_info/generate_random_pair.py
```

**Step 3 — Configure the truncation schedule**

Use an 𝐿 schedule L ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256} and enable the doubling-probe rule in your src/config/config.json.


**Step 4 — Run the large-graph experiment**

```bash
# Undirected graphs (e.g., Friendster)
python src/experiment/bd_for_large.py \
  --config src/config/config.json

# Directed graphs treated as undirected (e.g., Bluesky, Twitter-2010)
python src/experiment/bd_for_large.py \
  --config src/config/config.json \
  --dir2undir

```
Outputs are written under: data/paper/test/<dataset>/


##  3. Third-party baseline & data policy
- We do not redistribute third-party datasets. Please obtain them from the official sources listed above and follow the original licenses/terms.

- For PUSH/STW/SWF baselines, we only provide empty stubs (push.py, stw.py, swf.py). If you have legitimate access to the original implementations, place them in the expected locations and the runners will pick them up.


## 4. Anonymity & version freeze

- This artifact is anonymized and contains no identifying information.

- The repository is version-frozen for review (we will not update code during the review period).

- Post-acceptance, we will release a public, fully documented repository and (where applicable) deposit artifacts in an archival repository with a DOI.