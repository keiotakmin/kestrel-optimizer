#!/bin/bash
# 廃止 (2026-07-17): run_ictai_resume.sh に統合。
# 旧版は回帰の kestrel-cos を PROTO_PREFIX=protoe で走らせる設計で、
# Stage A の protoe_*/metrics.json を上書き破壊するバグがあった。
# kestrel-cos は resume 内で protoe3 / inrv3c / kodakc として走る。
echo "このスクリプトは廃止。experiments/run_ictai_resume.sh を使うこと" >&2
exit 1
