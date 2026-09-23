# actor
This repository contains the functionality of our toolkit for Outcome Review. To reproduce the method mentioned in the paper, here are the steps_
- Clone this repository.
```python
git clone https://github.com/mariavlachou/actor
cd actor
```
- Install the requirements. Please note that for the use of Pyterrier for retrieval (while running on a local Apple Silicon machine, the Anaconda version is used.  
```python
pip install -r requirements.txt
```
- To run the pipeline from getting data from the corresponding website up to producing the evaluation metrics, you can run:
 ```cd path/to/actor
/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace full_pipeline_example.ipynb --ExecutePreprocessor.timeout=1800

``` 
This file contains an example used for the fln dataset, and since the artifacts already exist, this will run in under a minute, saying that each step is already executed. On an Apple Silicon Mac, we use the Anaconda jupyter binary (the distribution where PyTerrier's JVM starts). When running on a cluster  or on a Linux machine, JDK + jupyter in your venv will work using
```jupyter nbconvert --to notebook --execute --inplace full_pipeline_example.ipynb --ExecutePreprocessor.timeout=1800
```
- If you would like to specify the dataset instead, you can run
