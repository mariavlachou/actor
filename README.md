# actor
This repository contains the functionality of ACTOR: our Asylum Chunking Toolkit for Outcome Review. To reproduce the method mentioned in the paper, here are the steps:
- Clone this repository.
```python
git clone https://github.com/mariavlachou/actor
cd actor
```
- Install the requirements. Please note that for the use of PyTerrier for retrieval (while running on a local Apple Silicon machine, the Anaconda version is used. To run PyTerrier, you need Java (JDK 11+) for PyTerrier's embedded JVM — module load java/openjdk, or conda install openjdk=21 (no Mac-specific workaround needed on Linux).
```python
pip install -r requirements.txt
```

## Running the pipeline with the precomputed values
- To run the pipeline from getting data from the corresponding website up to producing the evaluation metrics, you can run:
 ```cd path/to/actor
/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace full_pipeline_example.ipynb --ExecutePreprocessor.timeout=1800

``` 
This file contains an example used for the fln dataset, and since the artifacts already exist, this will run in under a minute, saying that each step is already executed. On an Apple Silicon Mac, we use the Anaconda jupyter binary (the distribution where PyTerrier's JVM starts). When running on a cluster  or on a Linux machine, JDK + jupyter in your venv will work using

 ```
jupyter nbconvert --to notebook --execute --inplace full_pipeline_example.ipynb --ExecutePreprocessor.timeout=1800
``` 

- If you would like to specify the dataset instead, you can run the file that allows you to select the type of website/dataset as:
```
/opt/anaconda3/bin/python3 src/full_pipeline_generic.py --dataset fln
```

or to specify how many samples you want

```
/opt/anaconda3/bin/python3 src/full_pipeline_generic.py --dataset euaa --sample-size 50
```


or to run it from a cluster follow the steps:
- Setup an environment:
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

and note that you need the following non-pip prerequisites:

Java (JDK 11+) for PyTerrier's embedded JVM — module load java/openjdk, or conda install openjdk=21 (no Mac-specific workaround needed on Linux).
Also, Ollama, running (ollama serve &) for example for models like gemma3:4b, llama3.1:8b etc.

- run the pipeline as
```
python3 src/full_pipeline_generic.py --dataset fln
```

Without the precomputed values, the full pipeline takes roughly 5-6 hours to run on an Apple M4
Pro with 24GB RAM. 

## Running the visualisation

To interact with the interface, you will need Streamlit (see in requirements). To run it locally, you can run:

```
streamlit run retrieval_results_app.py
```

or if you want to first kill an exiting instance, use

```
pkill -f "streamlit run retrieval_results_app.py"; sleep 1; cd path/to/actor && (streamlit run retrieval_results_app.py --server.port 8501 &) && sleep 3 && open http://localhost:8501

```

If you are trying to run it on a cluster, use: 

```
ssh -L 8501:localhost:8501 you@cluster-node "cd actor && source .venv/bin/activate && streamlit run retrieval_results_app.py --server.headless true --server.port 8501" & sleep 5 && open http://localhost:8501

```
