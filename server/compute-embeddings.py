import sys
from Bio import SeqIO
import numpy as np

embedder_id = sys.argv[1]
unique_dir = sys.argv[2]

embedder = None 
# XLNET 
if embedder_id=="xlnet":
    from bio_embeddings.embed.prottrans_xlnet_uniref100_embedder import ProtTransXLNetUniRef100Embedder
    embedder = ProtTransXLNetUniRef100Embedder()
# BERT 
if embedder_id=="bert":
    from bio_embeddings.embed.prottrans_bert_bfd_embedder import ProtTransBertBFDEmbedder
    embedder = ProtTransBertBFDEmbedder()
# ALBERT
if embedder_id=="albert":
    from bio_embeddings.embed.prottrans_albert_bfd_embedder import ProtTransAlbertBFDEmbedder
    embedder = ProtTransAlbertBFDEmbedder()
# ALBERT
if embedder_id=="onehot":
    from bio_embeddings.embed.one_hot_encoding_embedder import OneHotEncodingEmbedder
    embedder = OneHotEncodingEmbedder()
# T5
if embedder_id=="t5":
    from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5XLU50Embedder
    embedder = ProtTransT5XLU50Embedder(half_model=True)
# ESM1b
if embedder_id=="esm":
    from bio_embeddings.embed.esm_embedder import ESM1bEmbedder
    embedder = ESM1bEmbedder()

with open(unique_dir + '/fasta-file/sequences.fasta') as handle:
    for rec in SeqIO.parse(handle, "fasta"):
        embedding = None
        sequence = str(rec.seq) 
        if embedder_id == 'esm':
            treshold = 1022
            vectors = []
            while len(s) > 0:
                s1 = sequence[:treshold]
                sequence = sequence[treshold:]
                vectors1 = np.array(embedder.embed(s1))
                if len(vectors) > 0:
                    embedding = np.concatenate((vectors, vectors1))
                else:
                    embedding = vectors1
        else:
            embedding = embedder.embed(sequence)
        
        filename = f"{rec.id}.npy"
        np.save(f'{unique_dir}/generated-embeddings/{filename}', embedding)