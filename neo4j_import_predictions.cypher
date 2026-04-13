-- Copy alzheimers_top100_predictions.tsv and drkg_with_headers.tsv into the inport folder first

//Reset
MATCH (n) DETACH DELETE n;

DROP CONSTRAINT entity_drkg_id IF EXISTS;
DROP INDEX entity_type IF EXISTS;

//Load files
LOAD CSV WITH HEADERS FROM 'file:///alzheimers_top100_predictions.tsv' AS row
FIELDTERMINATOR '\t'
CALL (row) {
  MERGE (drug:PredictedDrug {drug_id: row.drug_id})
  SET drug.rank = toInteger(row.rank),
      drug.drug_name = row.drug_name,
      drug.prediction_score = toFloat(row.score),
      drug.entity_type = 'Compound'
} IN TRANSACTIONS OF 100 ROWS;


// Import relationships between entities
LOAD CSV WITH HEADERS FROM 'file:///drkg_with_headers.tsv' AS row
FIELDTERMINATOR '\t'
WITH row, connected_ids
WHERE row.source IN connected_ids 
  AND row.target IN connected_ids
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


// Connect predicted drugs to Alzheimer's disease
MATCH (drug:PredictedDrug)
MATCH (disease:Entity {drkg_id: 'Disease::MESH:D000544'})
MERGE (drug)-[r:PREDICTED_TREATMENT]->(disease)
SET r.score = drug.prediction_score,
    r.rank = drug.rank;


// Link to existing DRKG entities
MATCH (pred:PredictedDrug)
MATCH (entity:Entity)
WHERE entity.drkg_id = pred.drug_id
MERGE (pred)-[:SAME_AS]->(entity);


// VISUALIZATION QUERIES

// View top 10 predicted drugs
MATCH (drug:PredictedDrug)
WHERE drug.rank <= 10
RETURN drug
ORDER BY drug.rank;

// View all predicted drugs with their scores
MATCH (drug:PredictedDrug)
RETURN drug.rank as rank, 
       drug.drug_name as drug, 
       drug.prediction_score as score
ORDER BY rank;

// show how top drugs connect to Alzheimer's
MATCH path = (drug:PredictedDrug)-[:SAME_AS]->(:Entity)
             -[r:DRKG_REL*1..2]-
             (disease:Entity {drkg_id: 'Disease::MESH:D000544'})
WHERE drug.rank <= 20
RETURN path
LIMIT 100; //Increase or decrease to see more or less relations

// Show predicted drugs and their mechanisms (genes/pathways)
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
