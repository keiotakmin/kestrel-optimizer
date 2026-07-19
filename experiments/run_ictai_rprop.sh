#!/bin/bash
# Rprop ベースライン一式 (回帰 protoe5 / INR inrv3r / Kodak kodakr)
# Rprop ベースライン一式 (回帰 protoe5 / INR inrv3r / Kodak kodakr)
set -e
cd /auto/proj/fujimoto/grad/research/bachelor
PROTO_PREFIX=protoe5 PROTO_OPTS="rprop" \
    bash experiments/run_protocol.sh california concrete energy
for img in camera astronaut; do
    for seed in 42 43 44; do
        [ -f "results/inrv3r_${img}_s${seed}/metrics.json" ] && continue
        python experiments/pilot_inr.py --image "$img" --seed "$seed" \
            --steps 2000 --prefix inrv3r --optimizers rprop \
            --lrs 1e-4 3e-4 1e-3 3e-3 1e-2
    done
done
for i in $(seq -w 1 24); do
    [ -f "results/kodakr_kodim${i}_s42/metrics.json" ] && continue
    python experiments/pilot_inr.py --image "kodim$i" --seed 42 \
        --steps 2000 --prefix kodakr --optimizers rprop --lrs 3e-4 1e-3 3e-3
done
echo "RPROP DONE ($(date '+%F %T'))"
