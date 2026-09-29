# actor
This repository presents the functionality of ACTOR: our Asylum Chunking Toolkit for Outcome Review. To reproduce the method mentioned in the paper and to further explore our system, we provide further details below: 


## Table of Contents
- [About the System](#about-the-system)
- [Supported Website Types](#supported-website-types)
- [Getting Started](#getting-started)
- [Pipeline with precomputed values](#pipeline-with-precomputed-values)
  - [Full Pipeline](#full-pipeline)
  - [Run Retrieval](#run-retrieval)
  - [Replace Own Data](#replace-own-data)
- [Visualisation](#visualisation)
  - [Topics to Case Files](#topics-to-case-files)
  - [Case Files to Topics](#case-files-to-topics)
  

## About the System
ACTOR consists of two main parts:
- A pipeline that obtains data from a website and computes the values by chunking, performing topic extraction, and topic-based retrieval.
- A visualisation tool that uses the precomputed values and displays results based on a mapping between chunk ids and document ids. A demonstration video of how we visualise our system and results is available at https://youtu.be/y-tmlT778Zs/. 

The architecture of our pipeline can be seen below. First, entire documents, each corresponding to a case file of an application decision summary, are chunked into smaller pieces. We use Few-shot Query Generation to generate one query per chunk using the few-shot examples in /prompts/fewshot_examples.txt. Second, We use Topic Extraction to produce a clustered version of the underlying semantic information contained in the generated queries. We use BERTopic to obtain query embeddings, reduce their dimensions, and cluster the embeddings into topics. We name these topics using Gemma3. At Step 3, we use topic-based retrieval using the derived topic set and then a pool of relevance-judged chunks from various retrieval models for retrieval evaluation (to use as qrels). Then, any retrieval method can be used to produce metrics such as MAP, NDCG, etc. Finally, we visualise our results from topic-based retrieval by highlighting the relationship between a document id and the corresponding chunks. This is explained in the corresponding section. 
![Figure 1](images/actor_pipe.png)

Our system is primarily designed to run locally, since a main goal for its usage is to compare one's insights about asylum application/appeals that are often private and access is restricted. We provide a way to obtain data from openly available sources and then the user is free to compare locally with their own private data. Still, it can also be run from any machine. We now show how to get started if you want to use our system.

## Supported website types

The project supports two distinct website interaction patterns for scraping, plus one non-website data source (loading data from Hugging Face) — three Hugging Face datasets. See the details in the table below, where we show how we can obtain each type of data and which dataset is an example of the corresponding type. This is not an exhaustive list of how a dataset with asylum decision summary or appeals might look like, but we aim to provide a varied set of real options.

| Type | Module | Summary | Example site |
|---|---|---|---|
| Plain GET, URL-parameterized listing pages | `scraper.py` (`WebsiteScraper`) | Filters/pagination are plain URL query parameters, so a `requests` GET is enough. Generalized via a `SiteConfig` (or `--config site.json` for a new site). | fln.dk/praksis/ |
| JS/AJAX "callback" search pages | `browser_scraper.py` (`BrowserWebsiteScraper`) | No URL represents a filtered/paged state — filters and "load more" fire JS callbacks. Drives a real headless browser via Playwright instead. | caselaw.euaa.europa.eu |
| Pre-existing Hugging Face dataset | `hf_dataset_sampler.py` (`HFDatasetSampler`) | Not a scrape — samples directly from an already-published HF dataset via `datasets`. | `clairebarale/AsyLex` |


## Getting Started
- Clone this repository.
```python
git clone https://github.com/mariavlachou/actor
cd actor
```
- Install the requirements. Note that you will need to install PyTerrier for the retrieval step. If you are running on a local Apple Silicon machine, the Anaconda version of Python is used. To run PyTerrier, you need Java (JDK 11+) for PyTerrier's embedded JVM — module load java/openjdk, or conda install openjdk=21 (no Mac-specific workaround is needed if you use Linux).
```python
pip install -r requirements.txt
```

You will need Ollama if you have not already installed it (using ollama serve and ollama pull). We use 
```
ollama pull gemma3:4b
ollama pull hf.co/bartowski/google_gemma-3-4b-it-GGUF:Q5_K_S
ollama pull llama3.1:8b
ollama serve &
```


## Pipeline with precomputed values 
We now show how a user can obtain results from our pipeline. Without the precomputed values, the full pipeline takes roughly 5-6 hours to run on an Apple M4 Pro with 24GB RAM.

### Full Pipeline
- To run the full pipeline, which includes scraping data from the corresponding website up to producing the evaluation metrics, you need to run the file full_pipeline_generic.ipynb. We use
```
/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace full_pipeline_generic.ipynb
```
(using Anaconda's Python). --inplace writes all outputs back into the notebook itself, while at every stage, if anything is  already computed in data/ for the selected dataset, it skips it, so it runs fast for fln/euaa, where we have already computed them. 

To select another dataset or to specify a different sample size, you can use: 
```
/opt/anaconda3/bin/papermill full_pipeline_generic.ipynb output.ipynb \
    -p DATASET_KEY euaa \
    -p SAMPLE_SIZE 30 \
    -p SAMPLE_SEED 42
```
Currently, the available datasets options are fln, euaa, asylex.

If you are not running from Mac, any Python and JDK can work. Therefore, first set the environment:
```
module load java              # or: sudo apt-get install -y openjdk-17-jdk
pip install -r requirements.txt
playwright install chromium   # only needed if sourcing euaa from scratch
```
Again, you need an Ollama installation as before. 
Then, to import everything needed from src/, data/ and index_dir/, you can run:
```
rsync -avz full_pipeline_generic.ipynb src/ data/ index_dir/ user@cluster:/path/to/actor/
```
While data/ and index_dir/ are optional, bringing them lets step_done() check and skip anything that is already computed (instead of doing scraping/query-gen/judging from scratch.

Finally, run the pipeline using:
```
jupyter nbconvert --to notebook --execute --inplace full_pipeline_generic.ipynb
```

or with overrides using:
```
papermill full_pipeline_generic.ipynb output.ipynb -p DATASET_KEY euaa
```


The file full_pipeline_example.ipynb contains an example used for the fln dataset, and since the artifacts already exist, this will run in under a minute, saying that each step is already executed. On an Apple Silicon Mac, we use the Anaconda jupyter binary (the distribution where PyTerrier's JVM starts). When running on a cluster  or on a Linux machine, JDK + jupyter in your venv will work using

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

### Replace Own Data
You can use your own local data instead of our online examples to run the pipeline.

## Visualisation
A demonstration video of how we visualise our system and results is available at https://youtu.be/y-tmlT778Zs/. 

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

Below, we show examples of usage for each of the two main parts of the visualisation. 

### Topics to Case Files
First, we see first tab of our visualisation, where a user can select a dataset, a topic that is extracted from the dataset using our pipeline, and a retrieval method, and they can view for this specific selected topic, which case files answer it best.
![Figure 2](images/tab1_actor.png)

### Case Files to Topics
Then, we show an example of the second tab of our visualisation,
![Figure 3](images/tab2_actor.png)
