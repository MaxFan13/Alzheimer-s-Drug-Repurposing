#!/usr/bin/env python3
"""
Alzheimer's disease drug repurposing using DRKG TransE embeddings.
Ranks DrugBank compounds by predicted treatment score for AD.
"""
import csv

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as fn

from utils import download_and_extract

AD_DISEASE_CANDIDATES = [
    "Disease::MESH:D000544",
    "Disease::DOID:10652",
]

TREATMENT_RELATIONS = [
    "Hetionet::CtD::Compound:Disease",
    "GNBR::T::Compound:Disease",
]

GAMMA = 12.0
TOP_K = 100


def build_drug_list(entities_path, output_path):
    """Filter DrugBank compounds from the entity list and write them to a file.

    Args:
        entities_path: Path to the entities TSV file (name, id columns).
        output_path: Path where the filtered drug TSV will be written.

    Returns:
        List of drug entity name strings (e.g. 'Compound::DB00001').
    """
    entities = pd.read_csv(entities_path, sep='\t', names=['name', 'id'])
    drugs = entities[entities['name'].str.startswith('Compound::DB')]
    drugs.to_csv(output_path, sep='\t', header=False, index=False)

    drug_list = []
    with open(output_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t', fieldnames=['drug', 'ids']):
            drug_list.append(row['drug'])
    return drug_list


def load_entity_maps(entity_file, relation_file):
    """Load entity and relation name-to-id mappings from TSV files.

    Args:
        entity_file: Path to entities TSV (name, id columns).
        relation_file: Path to relations TSV (name, id columns).

    Returns:
        Tuple of (entity_map, entity_id_map, relation_map) where:
            entity_map: dict mapping entity name -> integer id.
            entity_id_map: dict mapping integer id -> entity name.
            relation_map: dict mapping relation name -> integer id.
    """
    entity_map = {}
    entity_id_map = {}
    relation_map = {}

    with open(entity_file, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t', fieldnames=['name', 'id']):
            entity_map[row['name']] = int(row['id'])
            entity_id_map[int(row['id'])] = row['name']

    with open(relation_file, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t', fieldnames=['name', 'id']):
            relation_map[row['name']] = int(row['id'])

    return entity_map, entity_id_map, relation_map


def resolve_disease_ids(candidates, entity_map):
    """Validate disease candidate identifiers against the entity map.

    Prints a warning for any candidates not found and raises if none match.

    Args:
        candidates: List of disease entity name strings to look up.
        entity_map: dict mapping entity name -> integer id.

    Returns:
        List of candidate names that exist in entity_map.

    Raises:
        ValueError: If none of the candidates are found in entity_map.
    """
    found = [d for d in candidates if d in entity_map]
    missing = set(candidates) - set(found)
    if missing:
        print("Not found in entities.tsv (skipped):", sorted(missing))
    if not found:
        raise ValueError("No Alzheimer disease nodes found. Update AD_DISEASE_CANDIDATES.")
    return found


def transE_l2(head, rel, tail):
    """Compute TransE L2 scores: gamma - ||head + rel - tail||_2.

    Args:
        head: Tensor of head entity embeddings.
        rel: Tensor of relation embeddings.
        tail: Tensor of tail entity embeddings.

    Returns:
        Tensor of scalar scores (higher is more plausible).
    """
    return GAMMA - th.norm(head + rel - tail, p=2, dim=-1)


def score_drugs(drug_emb, drug_ids, treatment_embs, disease_ids, entity_emb):
    """Score all drugs against each treatment relation and disease using TransE.

    Iterates over every (treatment relation, disease) pair, computes
    log-sigmoid TransE scores for all drugs, then concatenates and sorts
    results in descending order.

    Args:
        drug_emb: Tensor of shape (num_drugs, emb_dim) with drug embeddings.
        drug_ids: Tensor of integer drug entity ids (length num_drugs).
        treatment_embs: List of relation embedding tensors (one per treatment relation).
        disease_ids: List of integer disease entity ids.
        entity_emb: NumPy array of all entity embeddings indexed by id.

    Returns:
        Tuple of (scores, dids) as NumPy arrays sorted by descending score.
    """
    scores_list = []
    dids_list = []
    for treatment_emb in treatment_embs:
        for disease_id in disease_ids:
            disease_emb = entity_emb[disease_id]
            score = fn.logsigmoid(transE_l2(drug_emb, treatment_emb, disease_emb))
            scores_list.append(score)
            dids_list.append(drug_ids)

    scores = th.cat(scores_list)
    dids = th.cat(dids_list)
    idx = th.flip(th.argsort(scores), dims=[0])
    return scores[idx].numpy(), dids[idx].numpy()


def get_top_k(scores, dids, k):
    """Return the top-k unique drugs by first-occurrence rank.

    Because the same drug may be scored multiple times (once per
    relation/disease pair), this deduplicates by keeping the first
    (highest-ranked) occurrence of each drug id.

    Args:
        scores: NumPy array of scores sorted in descending order.
        dids: NumPy array of drug entity ids parallel to scores.
        k: Maximum number of unique drugs to return.

    Returns:
        Tuple of (top_dids, top_scores) NumPy arrays of length <= k.
    """
    _, unique_indices = np.unique(dids, return_index=True)
    top_indices = np.sort(unique_indices)[:k]
    return dids[top_indices], scores[top_indices]


def load_known_drugs(filepath):
    """Load a TSV of known Alzheimer's drugs into a DrugBank-id keyed dict.

    Args:
        filepath: Path to TSV with columns (id, drug_name, drug_id).

    Returns:
        dict mapping DrugBank id string -> drug name string.
    """
    known = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t', fieldnames=['id', 'drug_name', 'drug_id']):
            known[row['drug_id']] = row['drug_name']
    return known


def export_predictions(proposed_dids, proposed_scores, entity_id_map, output_file, topk):
    """Write the top-k drug predictions to a ranked TSV file.

    Args:
        proposed_dids: Array of top-k drug entity ids.
        proposed_scores: Array of corresponding scores.
        entity_id_map: dict mapping integer id -> entity name string.
        output_file: Path to the output TSV file.
        topk: Number of predictions to write.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("rank\tdrug_id\tdrug_name\tscore\n")
        for i in range(topk):
            drug_id = entity_id_map[int(proposed_dids[i])]
            drug_name = drug_id.split('::')[1] if '::' in drug_id else drug_id
            f.write(f"{i+1}\t{drug_id}\t{drug_name}\t{proposed_scores[i]}\n")


def add_drkg_headers(input_file, output_file):
    """Prepend a header row to the raw DRKG TSV and write to a new file.

    Args:
        input_file: Path to the raw DRKG TSV (no header).
        output_file: Path where the headered TSV will be written.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("source\trelation\ttarget\n")
        for line in infile:
            outfile.write(line)


def main():
    """Orchestrate the full drug repurposing pipeline.

    Downloads and extracts DRKG data, scores all DrugBank compounds against
    Alzheimer's disease nodes using TransE embeddings, prints the top-k
    results, highlights any known AD drugs in the top-k, and writes
    predictions to a TSV file.
    """
    download_and_extract()

    drug_list = build_drug_list('embed/entities.tsv', 'infer_drug.tsv')
    entity_map, entity_id_map, relation_map = load_entity_maps(
        'embed/entities.tsv', 'embed/relations.tsv'
    )

    disease_list = resolve_disease_ids(AD_DISEASE_CANDIDATES, entity_map)

    drug_ids = th.tensor([entity_map[d] for d in drug_list]).long()
    disease_ids = [entity_map[d] for d in disease_list]
    treatment_rid = th.tensor([relation_map[r] for r in TREATMENT_RELATIONS])

    entity_emb = np.load('embed/DRKG_TransE_l2_entity.npy')
    rel_emb = np.load('embed/DRKG_TransE_l2_relation.npy')

    drug_emb = th.tensor(entity_emb[drug_ids])
    treatment_embs = [th.tensor(rel_emb[rid]) for rid in treatment_rid]

    scores, dids = score_drugs(drug_emb, drug_ids, treatment_embs, disease_ids, entity_emb)
    proposed_dids, proposed_scores = get_top_k(scores, dids, TOP_K)

    for i in range(TOP_K):
        print("{}\t{}".format(entity_id_map[int(proposed_dids[i])], proposed_scores[i]))

    known_ad_map = load_known_drugs('alzheimers_known_drugs.tsv')
    print("\nKnown AD drugs in top {}:".format(TOP_K))
    for i in range(TOP_K):
        drug_db_id = entity_id_map[int(proposed_dids[i])][10:17]
        if drug_db_id in known_ad_map:
            print("[{}]\t{}\t{}".format(i, known_ad_map[drug_db_id], proposed_scores[i]))

    export_predictions(proposed_dids, proposed_scores, entity_id_map,
                       'alzheimers_top100_predictions.tsv', TOP_K)
    add_drkg_headers('drkg.tsv', 'drkg_with_headers.tsv')


if __name__ == '__main__':
    main()
