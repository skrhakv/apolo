# Apolo Server
## Instalation
Please see `install-bio-embeddings.sh` script for instalation steps for Ubuntu.

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