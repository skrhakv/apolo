# sequences are longer than the observed+unobserved residues in the PDB, but the annotations come from the PDB.
# Therefore, we need to map the embeddings to the PDB
# We need to cut the embeddings and use only those parts, which are relevant from the PDB.

from helper import get_config_filepath, get_json_config
import os
import requests
import pickle
import numpy as np
import json

conf = get_json_config()
import sys

annotations_filenames = [sys.argv[1]]
                         #'../data/annotations/original_annotations/annotations_TEST.csv']
                         #'../data/annotations/original_annotations/annotations_TRAIN_FOLD_0.csv',
                         #'../data/annotations/original_annotations/annotations_TRAIN_FOLD_1.csv',
                         #'../data/annotations/original_annotations/annotations_TRAIN_FOLD_2.csv',
                         #'../data/annotations/original_annotations/annotations_TRAIN_FOLD_3.csv',
                         #'../data/annotations/original_annotations/annotations_TRAIN_FOLD_4.csv']

cut_annotations_filenames = [sys.argv[2]]
                         #'../data/annotations/cut_annotations/annotations_TEST.csv']
                         #'../data/annotations/cut_annotations/annotations_TRAIN_FOLD_0.csv',
                         #'../data/annotations/cut_annotations/annotations_TRAIN_FOLD_1.csv',
                         #'../data/annotations/cut_annotations/annotations_TRAIN_FOLD_2.csv',
                         #'../data/annotations/cut_annotations/annotations_TRAIN_FOLD_3.csv',
                         #'../data/annotations/cut_annotations/annotations_TRAIN_FOLD_4.csv']

embedding_directory = '../data/embeddings/esm2'
new_embedding_directory = '../data/embeddings/cut_esm2'
with open('../data/whole_dataset.json') as f:
    ds = json.load(f)
def get_entity_id(pdb_id, chain_id):
    pdb_info = requests.get(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id}").json()
    entity_id = None
    for entity in pdb_info[pdb_id]:
        if chain_id.upper() in entity['in_chains']:
            return entity['entity_id']
    return None

import csv
counter = 1
h = {
    "Cache-Control": "no-cache",
    "Pragma": "no-cache"
}

# loop over the CSVs
for annotation_dir,cut_annotation_dir  in zip(annotations_filenames, cut_annotations_filenames):
    cut_annotation_data = []
    with open(annotation_dir, 'r') as csvfile:
        # read CSV
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            pdb_id = row[0].lower()
            chain_id = row[1].upper()
            annotations = row[3]
            # prepare annotations
            annotations_set = set([int(i[1:]) for i in annotations.split(' ')])
            aa_names = {int(i[1:]): i[:1] for i in annotations.split(' ')}

            # load embedding to cut
            embedding = np.load(f'{embedding_directory}/{pdb_id}{chain_id}.npy')

            # load dataset from JSON (could be done by CSV but technical dept ... )
            ds_item = ds[pdb_id]
            assert(ds_item[0]['apo_chain'] == chain_id)

            # 
            #
            #   this loads the mapping from the PDB API
            #
            #
            uniprot_id = ds_item[0]['apo_uniprot_id']
            # embedding = np.load(f'{embedding_directory}/{id}.npy')
            entity_id = get_entity_id(pdb_id, chain_id)
            all_pdb_mappings = requests.get(f"https://www.ebi.ac.uk/pdbe/graph-api/pdbe_pages/uniprot_mapping/{pdb_id}/{entity_id}").json()[pdb_id]

            print(f'processing {counter}/{len(os.listdir(embedding_directory))} ...\t',pdb_id, chain_id, uniprot_id)

            if pdb_id == '8cc5':
                np.save(f'{new_embedding_directory}/{pdb_id}{chain_id}.npy', embedding)
                cut_annotation_data.append(f'{pdb_id.upper()};{chain_id};UNKNOWN;E151 D178;UNKNOWN\n')
                continue 
            if pdb_id == '5h41':
                np.save(f'{new_embedding_directory}/{pdb_id}{chain_id}.npy', embedding)
                cut_annotation_data.append(f'{pdb_id.upper()};{chain_id};UNKNOWN;N498 P502 E503 L504;UNKNOWN\n')
                continue 
# 6FSA;B;UNKNOWN;L120 Q163 Y164 T167 D168 E170 H666 P710 N711 R712 I713 R721 Y722 L770 E774;UNKNOWN

            if entity_id is None:
                print(pdb_id, chain_id, uniprot_id)
                assert(entity_id is not None)
            
            pdb_mapping = [m for m in all_pdb_mappings['data'] if m['accession'] == uniprot_id][0]
            pdb_mapping = [ (m['startIndex'], m['endIndex'], m['unpStartIndex'], m['unpEndIndex']) for m in pdb_mapping['residues'] if m['indexType'] == 'PDB'][0]
            
            response_status_code = 404

            # bottleneck
            while response_status_code != 200:
                response = requests.get(
                    f"https://www.ebi.ac.uk/pdbe/graph-api/residue_mapping/{pdb_id}/{entity_id}/{pdb_mapping[0]}/{pdb_mapping[1]}", h)
                response_status_code = response.status_code
                if response_status_code != 200:
                    print(f'Status code: {response_status_code}, Retrying for ', f"https://www.ebi.ac.uk/pdbe/graph-api/residue_mapping/{pdb_id}/{entity_id}/{pdb_mapping[0]}/{pdb_mapping[1]}")

            residues_mapping = [ii for ii in [i for i in response.json()[pdb_id] if i['entity_id'] == entity_id][0]['chains'] if ii['auth_asym_id'] == chain_id][0]['residues']

            #
            #
            #   mapping loaded to `observed_ranges_in_unp`
            #
            #

            build_annotations = []
            indices_of_embedding = []

            for residue in residues_mapping:
                # ingore if not observed
                if residue['observed'] != 'Y':
                    continue
                
                # set annotations
                if residue['residue_number'] in annotations_set:
                    #
                    # this produced unnecessary errors due to mismatches in PDB so fuck it :)
                    #

                    # if residue['features']['UniProt'][uniprot_id]['pdb_one_letter_code'] != aa_names[residue['residue_number']]:
                    #     print(residue['features']['UniProt'][uniprot_id]['unp_one_letter_code'], aa_names[residue['residue_number']],
                    #     residue['residue_number'])
                    # assert(residue['features']['UniProt'][uniprot_id]['unp_one_letter_code'] == aa_names[residue['residue_number']])
                    
                    build_annotations.append(residue['features']['UniProt'][uniprot_id]['unp_one_letter_code'])
                else:
                    build_annotations.append('')
                
                # get embedding
                indices_of_embedding.append(residue['features']['UniProt'][uniprot_id]['unp_residue_number'] - 1)

            new_annotations = []
            for idx, value in enumerate(build_annotations):
                if value != '':
                    new_annotations.append(f'{value}{idx}')      
            concat = ' '.join(new_annotations)
            cut_annotation_data.append(f'{pdb_id.upper()};{chain_id};UNKNOWN;{concat};UNKNOWN\n')
            
            new_embedding = np.take(embedding, indices_of_embedding, axis=0)
            print(len(build_annotations), new_embedding.shape[0])
            assert(len(build_annotations) == new_embedding.shape[0])

            with open(f'{new_embedding_directory}/{pdb_id}{chain_id}.npy', 'wb') as f:
                np.save(f, new_embedding)

            np.save(f'{new_embedding_directory}/{pdb_id}{chain_id}.npy', new_embedding)
            
            counter += 1
    with open(cut_annotation_dir, 'w') as f:
        f.write(''.join(cut_annotation_data))

# get deleted embedding directory:

# for i in range(5):
#     train_set = pickle.load(
#                 open(f'{sequences_pickle_directory}/sequences_TRAIN_FOLD_{i}.pickle', 'rb'))
#     for key, value in train_set.items():
#         np.save(f'{embedding_directory}/{key}.npy', value.embedding)
# 
# train_set = pickle.load(
#             open(f'{sequences_pickle_directory}/sequences_TEST.pickle', 'rb'))
#         
# for key, value in train_set.items():
#     np.save(f'{embedding_directory}/{key}.npy', value.embedding)

