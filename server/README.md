# Apolo Server
## Instalation
Please see `install-bio-embeddings.sh` script for instalation steps for Ubuntu.

## Usage
If you have your server running, you can use cURL to send queries:
```
curl  -X  GET  -i  -F "fasta=@sequences.fasta" http://127.0.0.1:5000?embedder=bert --output tmp.zip
```
where you input URL to your server instead of `http://127.0.0.1:5000`. The parameter `embedder=bert` specifies the model and you should provide your protein sequences in the `sequences.fasta` file.

## Troubleshooting
I encountered this error during the loading of `ESM1bEmbedder`:
```
RuntimeError: unexpected EOF, expected 1542573126 more bytes. The file might be corrupted.
```
Solution was to delete the respective folder in `.cache` directory:
```
rm -rf ~/.cache/bio_embeddings/esm1b
```
Or (if you run as root):
```
sudo rm -rf /root/.cache/bio_embeddings/esm1b
```