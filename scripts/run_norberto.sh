#!/usr/bin/env bash
# NorBERTo no lugar do BERTimbau/mBERT, pipeline igual ao do trabalho anterior.
# Uso: nohup bash scripts/run_norberto.sh > logs/run_norberto.out 2>&1 &
# Teste de encanamento antes: SMOKE=1 bash scripts/run_norberto.sh
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
mkdir -p logs
LT="uv run --extra classification-embedding --extra classification python -m expanded_fake_news_corpus.classification"
M=norberto
SMOKE_FT=""; [ "${SMOKE:-0}" = 1 ] && SMOKE_FT="--limit-pairs 60 --folds-to-run 1 --epochs 1 --eval-steps 10 --warmup-steps 5 --label smoke_norberto"

run() {  # run <nome> <comando...>
  name=$1; shift
  echo "== $name: $(date)"
  "$@" > "logs/$name.log" 2>&1
  rc=$?
  echo "== $name: exit $rc em $(date)"
}

[ "${SMOKE:-0}" = 1 ] && { run smoke_ft $LT.finetune_transfer --bert-model $M $SMOKE_FT; exit 0; }

# 1) encoder congelado (mean pooling, padrão do NorBERTo)
run emb_$M  $LT.linguistic_transfer --mode embedding --encoder $M --classifier all --n-jobs 6
run comb_$M $LT.linguistic_transfer --mode combined  --encoder $M --classifier all --n-jobs 6
# 2) fine-tuning completo (5 folds, melhor por F1 de validação)
run ft_$M   $LT.finetune_transfer --bert-model $M
# 3) classificadores sobre o encoder ajustado (mean, não cls)
BEST=$(python3 -c "import json;print(json.load(open('data/classification/paper_replication/finetune_$M.json'))['best_model_dir'])")
run ftemb_$M $LT.linguistic_transfer --mode embedding --encoder "$BEST" --pooling mean \
  --classifier all --n-jobs 6 --label finetuned_embedding_$M
