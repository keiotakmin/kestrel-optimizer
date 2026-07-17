#!/bin/bash
# 廃止 (2026-07-17): run_ictai_resume.sh に統合。
# 旧版は PROTO_PREFIX=protoe で走る設計で、Stage A の protoe_*/metrics.json
# を上書き破壊するバグがあった (run_comparison は同名 dir に書く)。
# 旧 EAGLE ベースラインは resume 内で PROTO_PREFIX=protoe2 として走る。
echo "このスクリプトは廃止。experiments/run_ictai_resume.sh を使うこと" >&2
exit 1
