# Scholar Team Recommender UI Starter

This is a beginner-friendly starter project for a scholar recommendation UI that looks and behaves like a simple Google Scholar-inspired search page.

## What it does

- shows the user name on the top right of the webpage
- accepts a natural-language query in a text box
- recommends a team of scholars from NSF-funded PI dataset
- reads scholar information from JSON files
- optionally uses scholar embeddings from a `.npy` file
- falls back to lexical ranking if the query encoder and scholar embeddings are not in the same vector space

## Folder structure

```text
scholar_team_ui_starter/
├── app.py
├── prepare_data.py
├── build_embeddings.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
└── data/
    ├── nsf_json/
    ├── scholar_index.json        # created by prepare_data.py
    ├── scholar_ids.npy           # created by prepare_data.py or build_embeddings.py
    └── scholar_embeddings.npy    # created by build_embeddings.py or provided by you
```

## Step 1: Create a Python environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Add your JSON files

Put your 5,099 scholar JSON files inside:

```text
data/nsf_json/
```

## Step 4: Build the flattened scholar index

```bash
python prepare_data.py
```

This creates:

- `data/scholar_index.json`
- `data/scholar_ids.npy`

## Step 5A: If you do NOT already have embeddings

Run:

```bash
python build_embeddings.py
```

This creates:

- `data/scholar_embeddings.npy`
- `data/scholar_ids.npy`

## Step 5B: If you ALREADY have your own scholar embeddings

Place these files inside `data/`:

- `scholar_embeddings.npy`
- `scholar_ids.npy`

Important: `scholar_ids.npy` must store the scholar IDs in the same row order as your embeddings matrix.

## Step 6: Set your name for the top-right badge

Open `app.py` and change:

```python
USER_DISPLAY_NAME = "Your Name"
```

## Step 7: Run the web app

```bash
python app.py
```

Open this in your browser:

```text
http://127.0.0.1:5000
```

## Important note about embeddings

This starter app can compare the query embedding directly to the scholar embeddings only if:

1. the query encoder and scholar embeddings live in the same vector space, and
2. they have the same dimension.

The included `build_embeddings.py` uses:

- `sentence-transformers/all-MiniLM-L6-v2`

If your own `.npy` scholar embeddings were produced by a hypergraph model, you will usually need one extra component:

- either a `query -> scholar-embedding-space` projection model,
- or a hybrid scoring model that mixes lexical matching with graph-based signals.

The current UI is still useful even in that case, because it will fall back to lexical retrieval and display the same webpage.

## How to make it more advanced later

Once the basic UI is running, we can improve it by adding:

- scholar profile pages
- filtering by institution, program, or year
- better team diversity constraints
- a knowledge reasoning layer that converts the natural-language query into graph signals
- an authentication layer for real user accounts
- saving search history

