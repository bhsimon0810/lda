# Latent Dirichlet Allocation
A simple python implementation of LDA model with gibbs sampling and variational EM inference algorithms. The scripts have been tested on a small dataset, but it will take a long time to run on larger datasets since there aren't any computation optimization techniques.
## Usage
To just verify the inference algorithm:

```bash
python lda_gibbs_sampling.py

python lda_variational_inference.py
```

Then the topic-word distribution  will be exported as `csv` file.
