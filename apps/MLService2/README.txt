#add init files
ni apps\__init__.py -ItemType File -Force | Out-Null
ni apps\MLService2\__init__.py -ItemType File -Force | Out-Null

#docker commands
docker compose build mlservice2
docker compose up mlservice2
docker compose restart mlservice2
docker compose run --rm mlservice2 python app/training/scripts/prepare_data.py
docker compose run --rm mlservice2 python app/training/scripts/train_severity_hybrid.py
docker compose up -d mlservice2

#run locally
python app/training/scripts/prepare_data.py
python app/training/scripts/train_severity_hybrid.py

#site health
curl.exe http://localhost:8102/health

#test severity
$body = @{
  age = 67; hr = 120; sbp = 90; dbp = 60; rr = 28; temp = 39; cc_chestpain = 1
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8102/predict -ContentType 'application/json' -Body $body

#test with NLP
$triage = @{ text="Ngaitj koort kalyakal"; language="noongar" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8102/triage -ContentType 'application/json' -Body $triage