# sequences are longer than the observed+unobserved residues in the PDB, but the annotations come from the PDB.
# Therefore, we need to map the embeddings to the PDB
# We need to cut the embeddings and use only those parts, which are relevant from the PDB POI.


#
# The workflows goes as follows:
# first you need to obtain the entity id - you get it using the pdb id and chain id (docs: https://www.ebi.ac.uk/pdbe/api/doc):
# 
# https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/:pdbid
# 
# Next, you obtain the mapping from PDB to UniProt (docs: https://www.ebi.ac.uk/pdbe/graph-api/pdbe_doc/), 
# this retrieves the PDB start-end indices and their corresponding UniProt start-end indices:
# https://www.ebi.ac.uk/pdbe/graph-api/pdbe_pages/uniprot_mapping/:pdbId/:entityId
#
# For each residue in PDB structure, find out if it is observed or not, get its one-letter code and find its embedding in the embedding blob:
# https://www.ebi.ac.uk/pdbe/graph-api/residue_mapping/:pdbId/:entityId/:residueStart/:residueEnd
# 

import requests
import numpy as np
import csv

file = 'train-fold-3.csv'
INPUT_FILE = f'/home/skrhakv/apolo/data/cryptobench-annotations/{file}'
OUTPUT_FILE = f'/home/skrhakv/apolo/data/cryptobench-translated-annotations/{file}'
EMBEDDINGS_INPUT_DIR = '/home/skrhakv/esm2/embeddings/cryptobench-ahojv2'
EMBEDDINGS_OUTPUT_DIR = '/home/skrhakv/apolo/data/cryptobench-ahojv2-cut'
HEADER = {
    "Cache-Control": "no-cache",
    "Pragma": "no-cache"
}

def get_entity_id(pdb_id, chain_id):
    pdb_info = requests.get(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id}").json()
    for entity in pdb_info[pdb_id]:
        if chain_id in entity['in_chains']:
            return entity['entity_id']
    return None

def main():
    print(INPUT_FILE)
    with open(INPUT_FILE, 'r') as csvfile:
        counter = 0
        reader = csv.reader(csvfile, delimiter=';')
        skip = True
        for row in reader:
            pdb_id = row[0].lower()
            chain_ids = row[1].split('-')
            uniprot_ids = row[2].split('-')
            annotations = row[3].split(' ')

            if pdb_id == '7pxp':
                skip = False
            if skip: continue

            if len(uniprot_ids) < len(chain_ids):
                uniprot_ids = [uniprot_ids[0] for _ in chain_ids]
            
            elif len(uniprot_ids) > len(chain_ids): assert False, f'{pdb_id}; {chain_ids}; {uniprot_ids}'

            for uniprot_id, chain_id in zip(uniprot_ids, chain_ids):
                build_annotations = []
                indices_of_embedding = []

                chain_annotations = [i.split('_')[1] for i in annotations if i.split('_')[0] == chain_id]
                counter += 1
                print(f'processing {counter} ...\t',pdb_id, chain_id, uniprot_id, file)

                entity_id = get_entity_id(pdb_id, chain_id)
                assert entity_id is not None, f'{pdb_id}, {chain_id}, {uniprot_id}'

                all_pdb_mappings = requests.get(f"https://www.ebi.ac.uk/pdbe/graph-api/pdbe_pages/uniprot_mapping/{pdb_id}/{entity_id}").json()[pdb_id]
                # print(f"https://www.ebi.ac.uk/pdbe/graph-api/pdbe_pages/uniprot_mapping/{pdb_id}/{entity_id}")
                pdb_mapping = [m for m in all_pdb_mappings['data'] if m['accession'] == uniprot_id][0]
                pdb_mapping = [ (m['startIndex'], m['endIndex']) for m in pdb_mapping['residues'] if m['indexType'] == 'PDB'][0]
    
                response_status_code = 404
                
                while response_status_code != 200:
                    response = requests.get(
                        f"https://www.ebi.ac.uk/pdbe/graph-api/residue_mapping/{pdb_id}/{entity_id}/{pdb_mapping[0]}/{pdb_mapping[1]}", HEADER)
                    response_status_code = response.status_code
                    if response_status_code != 200:
                        print(f'Status code: {response_status_code}, Retrying for ', f"https://www.ebi.ac.uk/pdbe/graph-api/residue_mapping/{pdb_id}/{entity_id}/{pdb_mapping[0]}/{pdb_mapping[1]}")

                residues_mapping = [ii for ii in [i for i in response.json()[pdb_id] if i['entity_id'] == entity_id][0]['chains'] if ii['auth_asym_id'] == chain_id][0]['residues']

                old_build_annotations_len = len(build_annotations)
                for residue in residues_mapping:
                    # ingore if not observed
                    if residue['observed'] != 'Y':
                        continue
                    
                    # set annotations
                    if str(residue['author_residue_number']) in chain_annotations:
                        #
                        # this produced unnecessary errors due to mismatches in PDB so fuck it :)
                        #

                        # if residue['features']['UniProt'][uniprot_id]['pdb_one_letter_code'] != aa_names[residue['residue_number']]:
                        #     print(residue['features']['UniProt'][uniprot_id]['unp_one_letter_code'], aa_names[residue['residue_number']],
                        #     residue['residue_number'])
                        # assert(residue['features']['UniProt'][uniprot_id]['unp_one_letter_code'] == aa_names[residue['residue_number']])

                        build_annotations.append(f"{chain_id}_{residue['features']['UniProt'][uniprot_id]['unp_one_letter_code']}")
                    else:
                        build_annotations.append('')

                    # get embedding
                    indices_of_embedding.append(residue['features']['UniProt'][uniprot_id]['unp_residue_number'] - 1)
                
                embedding = np.load(f'{EMBEDDINGS_INPUT_DIR}/{uniprot_id}.npy')

                new_embedding = np.take(embedding, indices_of_embedding, axis=0)

                assert len(build_annotations) - old_build_annotations_len == new_embedding.shape[0], f'{len(build_annotations)}, {new_embedding.shape[0]}'
    
                with open(f'{EMBEDDINGS_OUTPUT_DIR}/{pdb_id}{chain_id}.npy', 'wb') as f:
                    np.save(f, new_embedding)
            
                new_annotations = []

                for idx, value in enumerate(build_annotations):
                    if value != '':
                        new_annotations.append(f'{value}{idx}')      
                concat = ' '.join(new_annotations)
                new_annotations = f'{pdb_id};{chain_id};{uniprot_id};{concat};UNKNOWN\n'
                with open(OUTPUT_FILE, 'a') as f:
                    f.write(new_annotations)

if __name__ == '__main__':
    main()