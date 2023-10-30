from flask import Flask, request, after_this_request, send_file, abort
import subprocess
import uuid
import os
import shutil

app = Flask(__name__)

# debug using curl:
# curl  -X  GET  -i  -F "fasta=@sequences.fasta" http://127.0.0.1:5000?embedder=bert --output tmp.zip
# where 'sequences.fasta' is a FASTA file containing sequences in FASTA format
@app.route('/')
def compute_embeddings():

    # generate output dir
    unique_dir = uuid.uuid4().hex
    fasta_dir = unique_dir + '/fasta-file'
    embeddings_dir = unique_dir + '/generated-embeddings'

    os.makedirs(unique_dir)
    os.makedirs(fasta_dir)
    os.makedirs(embeddings_dir)

    # define what to do after the request has finished
    @after_this_request
    def remove_file(response):
        try:
            shutil.rmtree(unique_dir)
        except Exception as error:
            app.logger.error("Error removing generated directory", error)
        return response
    
    # read the embedder type
    embedder = request.args.get('embedder', default=-1)
    
    if embedder not in ['bert', 't5', 'esm']:
        abort(400, 'Unknown embedder type') 

    # load the sequences into a file
    f = request.files['fasta']
    f.save(fasta_dir + '/sequences.fasta')

    # compute the embeddings
    subprocess.call(['sh', './compute-embeddings.sh', embedder, unique_dir])
    
    # create an ZIP archive and send it as a response
    archive_path = f'{unique_dir}/embeddings'

    shutil.make_archive(root_dir=embeddings_dir, format='zip', base_name=archive_path)
    return send_file(f'{archive_path}.zip')

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)

