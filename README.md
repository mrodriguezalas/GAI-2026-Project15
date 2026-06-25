# GAI-2026-Project15

## Install dependencies
- Conda environment: `conda create --name gai26 python=3.12`
- Activate the environment: `conda activate gai26`
- Install dependencies: `pip install -r requirements.txt`

## Create the PDF embedding
- Run the script: `python create_embedding.py`. This will create a `pdf_chunks.index` and `pdf_metadata.npy` files in the src/ directory.

## Run the app
- Run the app: `streamlit run app.py`