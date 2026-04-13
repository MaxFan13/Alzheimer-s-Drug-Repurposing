#!/usr/bin/env python3
"""
Alzheimer's Disease Drug Repurposing
Ranks candidate compounds with the pretrained DRKG TransE model
"""


# ============================================================================
# Alzheimer’s disease drug repurposing via disease–compound relations
# Ranks candidate compounds with the pretrained DRKG TransE model. For graph storage and Cypher, copy `drkg.tsv` into Neo4j’s import folder and run the scripts in `../neo4j/` (see `drug_repurpose/Readme.md`).
# ============================================================================


# ============================================================================
# Alzheimer’s disease targets in DRKG
# We use standard DRKG disease strings (often MeSH or DOID). After `entities.tsv` is available, we keep only IDs that exist in the graph. Add or remove candidates in `AD_disease_candidates` (search `entities.tsv` for “Alzheimer” if you want related phenotypes).
# ============================================================================

# Populated after entity_map is loaded (next section after download)
AD_disease_candidates = [
    "Disease::MESH:D000544",  # Alzheimer disease (MeSH)
    "Disease::DOID:10652",   # Alzheimer disease (DOID), if present
]
AD_disease_list = []  # set in embedding prep cell


# ============================================================================
# Candidate drugs
# Now we use FDA-approved drugs in Drugbank as candidate drugs. (we exclude drugs with molecule weight < 250) The drug list is in infer\_drug.tsv
# ============================================================================


# ============================================================================
# Treatment relation
# ============================================================================


# ============================================================================
# Two treatment relations in this context
# ============================================================================

treatment = ['Hetionet::CtD::Compound:Disease','GNBR::T::Compound:Disease']


# ============================================================================
# Get pretrained model
# We can directly use the pretrianed model to do drug repurposing.
# ============================================================================

import pandas as pd
import numpy as np
import os
import urllib.request
import tarfile

# ============================================================================
# Download and extract DRKG embeddings (standalone function)
# ============================================================================

def download_and_extract():
    """Download DRKG embeddings if not already present"""
    
    # Create embed directory if it doesn't exist
    os.makedirs('embed', exist_ok=True)
    
    # Check if files already exist
    required_files = [
        'embed/entities.tsv',
        'embed/relations.tsv',
        'embed/DRKG_TransE_l2_entity.npy',
        'embed/DRKG_TransE_l2_relation.npy'
    ]
    
    if all(os.path.exists(f) for f in required_files):
        print("✓ DRKG embeddings already downloaded")
        return
    
    print("Downloading DRKG embeddings...")
    print("NOTE: This is a ~200MB download and may take several minutes")
    
    # Try multiple URLs
    urls = [
        "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/embed.tar.gz",
        "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/DRKG/embed.tar.gz"
    ]
    
    tar_file = "embed.tar.gz"
    success = False
    
    for url in urls:
        try:
            print(f"Trying: {url}")
            # Add headers to avoid 403 error
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            with urllib.request.urlopen(req) as response, open(tar_file, 'wb') as out_file:
                # Download in chunks with progress
                file_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress every 10MB
                    if downloaded % (10 * 1024 * 1024) == 0:
                        mb_downloaded = downloaded / (1024 * 1024)
                        print(f"  Downloaded: {mb_downloaded:.1f} MB")
            
            print("✓ Download complete")
            success = True
            break
            
        except Exception as e:
            print(f"  Failed: {e}")
            if os.path.exists(tar_file):
                os.remove(tar_file)
            continue
    
    if not success:
        print("\n" + "="*80)
        print("ERROR: Could not download DRKG embeddings automatically")
        print("="*80)
        print("\nPlease download manually:")
        print("1. Visit: https://github.com/gnn4dr/DRKG")
        print("2. Download 'embed.tar.gz' from the releases")
        print("3. Place it in this directory")
        print("4. Run this script again")
        print("\nOr extract manually and place these files in 'embed/' folder:")
        for f in required_files:
            print(f"  - {f}")
        print("="*80)
        raise Exception("Failed to download DRKG embeddings")
    
    # Extract
    print("Extracting files...")
    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall('.')
        print("✓ Extraction complete")
    except Exception as e:
        print(f"ERROR extracting: {e}")
        raise
    
    # Clean up
    if os.path.exists(tar_file):
        os.remove(tar_file)
        print("✓ Cleaned up temporary files")

# Download embeddings
download_and_extract()

entity_idmap_file = 'embed/entities.tsv'
relation_idmap_file = 'embed/relations.tsv'


# ============================================================================
# Get embeddings for diseases and drugs
# ============================================================================

import pandas as pd

# Load entities from the embed folder
entities = pd.read_csv('embed/entities.tsv', sep='\t', names=['name', 'id'])

# Filter for compounds only (DrugBank drugs)
drugs = entities[entities['name'].str.startswith('Compound::DB')]

# Filter by molecular weight > 250 (as mentioned in notebook)
# For now, just export all DrugBank compounds
drugs.to_csv('infer_drug.tsv', sep='\t', header=False, index=False)

import csv

# Load entity file
drug_list = []
with open("infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug','ids'])
    for row_val in reader:
        drug_list.append(row_val['drug'])

# Get drugname/disease name to entity ID mappings
entity_map = {}
entity_id_map = {}
relation_map = {}
with open(entity_idmap_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
    for row_val in reader:
        entity_map[row_val['name']] = int(row_val['id'])
        entity_id_map[int(row_val['id'])] = row_val['name']
        
with open(relation_idmap_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
    for row_val in reader:
        relation_map[row_val['name']] = int(row_val['id'])
        
# handle the ID mapping
AD_disease_list = [d for d in AD_disease_candidates if d in entity_map]
missing = set(AD_disease_candidates) - set(AD_disease_list)
if missing:
    print("Not found in entities.tsv (skipped):", sorted(missing))
if not AD_disease_list:
    raise ValueError("No Alzheimer disease nodes found; search entities.tsv and update AD_disease_candidates.")

drug_ids = []
disease_ids = []
for drug in drug_list:
    drug_ids.append(entity_map[drug])
    
for disease in AD_disease_list:
    disease_ids.append(entity_map[disease])

treatment_rid = [relation_map[treat]  for treat in treatment]

# Load embeddings
import torch as th
entity_emb = np.load('embed/DRKG_TransE_l2_entity.npy')
rel_emb = np.load('embed/DRKG_TransE_l2_relation.npy')

drug_ids = th.tensor(drug_ids).long()
disease_ids = th.tensor(disease_ids).long()
treatment_rid = th.tensor(treatment_rid)

drug_emb = th.tensor(entity_emb[drug_ids])
treatment_embs = [th.tensor(rel_emb[rid]) for rid in treatment_rid]


# ============================================================================
# Drug Repurposing Based on Edge Score
# We use following algorithm to calculate the edge score. Note, here we use logsigmiod to make all scores < 0. The larger the score is, the stronger the $h$ will have $r$ with $t$.
# $\mathbf{d} = \gamma - ||\mathbf{h}+\mathbf{r}-\mathbf{t}||_{2}$
# $\mathbf{score} = \log\left(\frac{1}{1+\exp(\mathbf{-d})}\right)$
# When doing drug repurposing, we only use the treatment related relations.
# ============================================================================

import torch.nn.functional as fn

gamma=12.0
def transE_l2(head, rel, tail):
    score = head + rel - tail
    return gamma - th.norm(score, p=2, dim=-1)

scores_per_disease = []
dids = []
for rid in range(len(treatment_embs)):
    treatment_emb=treatment_embs[rid]
    for disease_id in disease_ids:
        disease_emb = entity_emb[disease_id]
        score = fn.logsigmoid(transE_l2(drug_emb, treatment_emb, disease_emb))
        scores_per_disease.append(score)
        dids.append(drug_ids)
scores = th.cat(scores_per_disease)
dids = th.cat(dids)


# sort scores in decending order
idx = th.flip(th.argsort(scores), dims=[0])
scores = scores[idx].numpy()
dids = dids[idx].numpy()


# ============================================================================
# Now we output proposed treatments
# ============================================================================

_, unique_indices = np.unique(dids, return_index=True)
topk=100
topk_indices = np.sort(unique_indices)[:topk]
proposed_dids = dids[topk_indices]
proposed_scores = scores[topk_indices]


# ============================================================================
# Now we list the pairs of in form of (drug, treat, disease, score)
# We select top K relevent drugs according the edge score
# ============================================================================

for i in range(topk):
    drug = int(proposed_dids[i])
    score = proposed_scores[i]
    
    print("{}\t{}".format(entity_id_map[drug], score))


# ============================================================================
# Check known symptomatic AD drugs (optional sanity check)
# A short list of common symptomatic therapies (DrugBank IDs) in `alzheimers_known_drugs.tsv`. Overlap with top‑K predictions is informative but not a gold standard—many effective compounds may not be labeled for AD in DRKG.
# ============================================================================

known_ad_file = 'alzheimers_known_drugs.tsv'
known_ad_map = {}
with open(known_ad_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'drug_name','drug_id'])
    for row_val in reader:
        known_ad_map[row_val['drug_id']] = row_val['drug_name']
        
for i in range(topk):
    drug = entity_id_map[int(proposed_dids[i])][10:17]
    if known_ad_map.get(drug, None) is not None:
        score = proposed_scores[i]
        print("[{}]\t{}\t{}".format(i, known_ad_map[drug], score))

# Export top 100 predictions to TSV for Neo4j import
output_file = 'alzheimers_top100_predictions.tsv'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("rank\tdrug_id\tdrug_name\tscore\n")
    for i in range(topk):
        drug = int(proposed_dids[i])
        score = proposed_scores[i]
        drug_id = entity_id_map[drug]
        # Extract DrugBank ID (e.g., DB00843 from Compound::DB00843)
        drug_name = drug_id.split('::')[1] if '::' in drug_id else drug_id
        f.write(f"{i+1}\t{drug_id}\t{drug_name}\t{score}\n")
        
print(f"✓ Saved top {topk} predictions to {output_file}")
print(f"  File ready for Neo4j import!")

len(known_ad_map)
