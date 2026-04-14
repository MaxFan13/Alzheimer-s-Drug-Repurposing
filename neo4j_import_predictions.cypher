// Copy alzheimers_top100_predictions.tsv and drkg_with_headers.tsv into the import folder first

// Reset
MATCH (n) DETACH DELETE n;

DROP CONSTRAINT entity_drkg_id IF EXISTS;
DROP INDEX entity_type IF EXISTS;

// Load predicted drugs
LOAD CSV WITH HEADERS FROM 'file:///alzheimers_top100_predictions.tsv' AS row
FIELDTERMINATOR '\t'
CALL (row) {
  MERGE (drug:PredictedDrug {drug_id: row.drug_id})
  SET drug.rank = toInteger(row.rank),
      drug.drug_name = row.drug_name,
      drug.prediction_score = toFloat(row.score),
      drug.entity_type = 'Compound'
} IN TRANSACTIONS OF 100 ROWS;


// Import DRKG relationships involving predicted drugs
MATCH (pred:PredictedDrug)
WITH collect(pred.drug_id) AS connected_ids
LOAD CSV WITH HEADERS FROM 'file:///drkg_with_headers.tsv' AS row
FIELDTERMINATOR '\t'
WITH row, connected_ids
WHERE (row.source IN connected_ids OR row.target IN connected_ids)
  AND row.source IS NOT NULL
  AND row.relation IS NOT NULL
  AND row.target IS NOT NULL
CALL (row) {
  WITH row
  MERGE (a:Entity {drkg_id: row.source})
  SET a.entity_type = split(row.source, '::')[0]
  MERGE (b:Entity {drkg_id: row.target})
  SET b.entity_type = split(row.target, '::')[0]
  MERGE (a)-[r:DRKG_REL {relation: row.relation}]->(b)
} IN TRANSACTIONS OF 5000 ROWS;


// Connect predicted drugs to Alzheimer's disease node
MATCH (drug:PredictedDrug)
MATCH (disease:Entity {drkg_id: 'Disease::MESH:D000544'})
MERGE (drug)-[r:PREDICTED_TREATMENT]->(disease)
SET r.score = drug.prediction_score,
    r.rank = drug.rank;


// Link predicted drugs to their DRKG entity counterparts
MATCH (pred:PredictedDrug)
MATCH (entity:Entity)
WHERE entity.drkg_id = pred.drug_id
MERGE (pred)-[:SAME_AS]->(entity);


// --- Visualization queries ---

// Top 10 predicted drugs
MATCH (drug:PredictedDrug)
WHERE drug.rank <= 10
RETURN drug
ORDER BY drug.rank;

// All predicted drugs with scores
MATCH (drug:PredictedDrug)
RETURN drug.rank as rank,
       drug.drug_name as drug,
       drug.prediction_score as score
ORDER BY rank;

// How top drugs connect to Alzheimer's (1-2 hops)
MATCH path = (drug:PredictedDrug)-[:SAME_AS]->(:Entity)
             -[r:DRKG_REL*1..2]-
             (disease:Entity {drkg_id: 'Disease::MESH:D000544'})
WHERE drug.rank <= 20
RETURN path
LIMIT 100;

// Top drugs with gene/pathway intermediates
MATCH (drug:PredictedDrug)-[:SAME_AS]->(compound:Entity)
      -[:DRKG_REL]-(intermediate:Entity)
      -[:DRKG_REL]-(disease:Entity {drkg_id: 'Disease::MESH:D000544'})
WHERE drug.rank <= 30
  AND intermediate.entity_type IN ['Gene', 'Pathway']
RETURN drug.rank as rank,
       drug.drug_name as drug,
       drug.prediction_score as score,
       intermediate.entity_type as mechanism_type,
       intermediate.drkg_id as mechanism
ORDER BY rank
LIMIT 50;


// Drugs sharing the most gene targets (similar mechanisms)
MATCH (drug1:PredictedDrug)-[:SAME_AS]->(:Entity)-[:DRKG_REL]-(gene:Entity {entity_type: 'Gene'})-[:DRKG_REL]-(:Entity)<-[:SAME_AS]-(drug2:PredictedDrug)
WHERE drug1.rank <= 20 AND drug2.rank <= 20 AND id(drug1) < id(drug2)
WITH drug1, drug2, count(DISTINCT gene) as shared_genes
WHERE shared_genes >= 3
RETURN drug1.drug_name as drug_1,
       drug1.rank as rank_1,
       drug2.drug_name as drug_2,
       drug2.rank as rank_2,
       shared_genes
ORDER BY shared_genes DESC
LIMIT 20;


// Hub genes connected to many top-ranked drugs
MATCH (drug:PredictedDrug)-[:SAME_AS]->(:Entity)-[:DRKG_REL]-(gene:Entity {entity_type: 'Gene'})
WHERE drug.rank <= 20
WITH gene, count(DISTINCT drug) as drug_count, collect(drug.drug_name) as drugs
WHERE drug_count >= 3
RETURN gene.drkg_id as gene,
       drug_count as connected_drugs,
       drugs[0..5] as sample_drugs
ORDER BY drug_count DESC
LIMIT 20;


// Relationship types connecting top drugs to Alzheimer's
MATCH (drug:PredictedDrug)-[:SAME_AS]->(:Entity)-[r:DRKG_REL]-(disease:Entity {drkg_id: 'Disease::MESH:D000544'})
WHERE drug.rank <= 20
WITH r.relation as relation_type, count(*) as frequency, collect(DISTINCT drug.drug_name) as drugs
RETURN relation_type,
       frequency,
       drugs[0..5] as sample_drugs
ORDER BY frequency DESC;


// Top-ranked drugs not in the known AD drug list (novel candidates)
MATCH (drug:PredictedDrug)
WHERE drug.rank <= 30
  AND NOT drug.drug_name IN ['DB00843', 'DB00674', 'DB00989', 'DB01043', 'DB00382']
RETURN drug.rank as rank,
       drug.drug_name as novel_candidate,
       drug.prediction_score as score
ORDER BY rank;
