-- ============================================================================
-- NEO4J IMPORT QUERIES FOR ALZHEIMER'S DRUG PREDICTIONS
-- ============================================================================

-- STEP 1: Import the top 100 predicted drugs
-- Copy alzheimers_top100_predictions.tsv to Neo4j's import folder first!

LOAD CSV WITH HEADERS FROM 'file:///alzheimers_top100_predictions.tsv' AS row
FIELDTERMINATOR '\t'
CALL {
  WITH row
  MERGE (drug:PredictedDrug {drug_id: row.drug_id})
  SET drug.rank = toInteger(row.rank),
      drug.drug_name = row.drug_name,
      drug.prediction_score = toFloat(row.score),
      drug.entity_type = 'Compound'
} IN TRANSACTIONS OF 100 ROWS;

-- ============================================================================

-- STEP 2: Connect predicted drugs to Alzheimer's disease (if you imported drkg data)
MATCH (drug:PredictedDrug)
MATCH (disease:Entity {drkg_id: 'Disease::MESH:D000544'})
MERGE (drug)-[r:PREDICTED_TREATMENT]->(disease)
SET r.score = drug.prediction_score,
    r.rank = drug.rank;

-- ============================================================================

-- STEP 3: Link to existing DRKG entities (if you imported drkg_with_headers.tsv)
MATCH (pred:PredictedDrug)
MATCH (entity:Entity)
WHERE entity.drkg_id = pred.drug_id
MERGE (pred)-[:SAME_AS]->(entity);

-- ============================================================================

-- VISUALIZATION QUERIES
-- ============================================================================

-- View top 10 predicted drugs
MATCH (drug:PredictedDrug)
WHERE drug.rank <= 10
RETURN drug
ORDER BY drug.rank;

-- View all predicted drugs with their scores
MATCH (drug:PredictedDrug)
RETURN drug.rank as rank, 
       drug.drug_name as drug, 
       drug.prediction_score as score
ORDER BY rank;

-- If you imported DRKG data, show how top drugs connect to Alzheimer's
MATCH path = (drug:PredictedDrug)-[:SAME_AS]->(:Entity)
             -[r:DRKG_REL*1..2]-
             (disease:Entity {drkg_id: 'Disease::MESH:D000544'})
WHERE drug.rank <= 20
RETURN path
LIMIT 100;

-- Show predicted drugs and their mechanisms (genes/pathways)
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
