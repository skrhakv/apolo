# APOLO
APOLO is a framework designed to create a simple neural network for predicting binding sites, using protein language model (*pLM*) embeddings as input. The framework leverages `SklearnTuner` from the `keras_tuner` library to perform hyperparameter tuning (e.g., number of layers, neurons per layer, dropout rate) using a predefined K-fold training split. After identifying the optimal parameters, the network is trained on the entire training set and its performance evaluated on a test set.

## Data
To run the framework, you need to supply pLM embeddings and annotations indicating which residues are binding (positive examples) or non-binding (negative examples). Each structure with `M` residues has embedding represented as an `MxN` numpy array, where `N` is the dimension of the pLM embeddings. The path to these data files should be specified in the `configuration.json` file.

### Annotation format
Annotations should follow this format:
```
{PDB-ID};{CHAIN-ID};{UNIPROT-ID};{ANNOTATIONS};{SEQUENCE}
```
While the `UNIPROT-ID` and `SEQUENCE` fields are included, they are not currently used by the framework and can contain arbitrary values.

For example, the annotation for structure `7qoqA` could look like this:
```
7qoq;A;K5B7Z4;A_T240 A_D241 A_A243 A_A244 A_L266 A_T267 A_Q268 A_R271 A_T272 A_S273;UNKNOWN
```
The `ANNOTATIONS` field lists binding residues, separated by spaces, in the format `{CHAIN-ID}_{AMINO-ACID-LETTER}{INDEX-OF-AMINOACID}`, where `INDEX-OF-AMINOACID` corresponds to the zero-based index for the residue position in the embedding array. For instance, `A_T240` indicates a binding residue at the 240th position in the embedding array (e.g., `embedding[240]` in Python). 


## Running the Framework
To execute APOLO, use the command:
```bash
python3 main.py
```
If you're running the framework on a cluster with SLURM, submit the job with:
```bash
sbatch --gpus=1 run-sbatch.sh
```
