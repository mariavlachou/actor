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
