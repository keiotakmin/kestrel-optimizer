#!/bin/bash
# K / β_h 感度走査 (論文 Table III の生成元、prefix=protoe4)
# K / β_h 感度走査 (回帰 3 データセット、prefix=protoe4)
set -e
cd /auto/proj/fujimoto/grad/research/bachelor
PROTO_PREFIX=protoe4 \
PROTO_OPTS="eagle-dqn-cd-k5 eagle-dqn-cd-k80 eagle-dqn-cd-bh05 eagle-dqn-cd-bh095" \
    bash experiments/run_protocol.sh california concrete energy
echo "SENS SWEEP DONE ($(date '+%F %T'))"
