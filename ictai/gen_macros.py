"""ICTAI 2026 論文の macros.tex を実験データから生成する (single-source rule)。

参照実装 (eagle/experiments/tsf_edge/gen_macros.py) と同じ思想:
本文の結果数値は全てここで生成されたマクロ経由で参照し、手打ちしない。
実験の再走後は sync_assets.sh (= 本スクリプト + 図の再生成) を回す。

データ源:
- 回帰      : results/protoe_* + protoeh_* (Stage A, run_ictai_baselines.sh)
- アブレーション: results/protoa_* + protob_* + protod_*
- INR       : results/inrv3_*   (Stage B)
- Kodak-24  : results/kodakb_*  (Stage C)
未着のデータブロックは TBD マクロ (赤字) になる。

実行: python ictai/gen_macros.py   (bachelor/ ルートから)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "paper" / "macros.tex"

sys.path.insert(0, str(ROOT / "experiments"))
from analyze_protocol import collect, x_to  # noqa: E402

TBD = r"\textcolor{red}{\textbf{TBD}}"
SEEDS = (42, 43, 44)
DS_TOK = {"california": "Cal", "concrete": "Con", "energy": "Ene"}
FAM_TOK = {
    "adam": "Adam", "adamw": "Adamw", "adam-cos": "Adamcos",
    "adabelief": "Adabelief", "adahessian": "Adahessian",
    "bb": "Bb", "bb-stab": "Bbstab", "lbfgs": "Lbfgs",
    "rprop": "Rprop",
    "eagle3": "Eaglethree", "eagle-dqn-cd": "Eagle",
    # 旧 EAGLE (arXiv:2502.01036) の正確な構成 = CLI "eagle"
    # (適応閾値切替 + lr 付き割線)。run_ictai_ext2.sh で protoe に追加
    "eagle": "Eaglearxiv",
    # KESTREL + cosine フォールバック (run_ictai_ext3.sh で inrv3c/kodakc)
    "kestrel-cos": "Kestrelcos",
    # アブレーション変種
    "eagle-orig": "Eagleorig", "eagle-dqn": "Eagledqn",
    "eagle3-noins": "Eagleswitchonly", "eagle4-aj": "Eagleaj",
    "eagle4-noins": "Eaglenoins", "eagle4": "Eaglefour",
    "eagle5": "Eaglefive", "eagle4-bh0": "Eaglebhzero",
    "eagle4-aj-bh0": "Eagleajbhzero", "eagle-dqn-bh0": "Eagledqnbhzero",
    "eagle4-aj-noins": "Eagleajnoins",
}
FRAC_TOK = {0.5: "Early", 0.8: "Mid", 0.9: "Late", 0.95: "Final"}

macros = {}


def emit(name, value):
    if name in macros:
        raise ValueError(f"duplicate macro: {name}")
    macros[name] = value


def fmt_ratio(vals, n_expect, prec=2):
    """速度比 mean±sd。部分到達は上付き (k/n)、全滅は ---。"""
    ok = [v for v in vals if v is not None]
    if not ok:
        return "---"
    m, s = np.mean(ok), np.std(ok)
    if len(ok) < n_expect:
        return f"{m:.{prec}f}$^{{({len(ok)}/{n_expect})}}$"
    return f"{m:.{prec}f}$\\pm${s:.{prec}f}"


def fmt_val(vals, prec=4):
    ok = [v for v in vals if v is not None]
    if not ok:
        return "---"
    return f"{np.mean(ok):.{prec}f}$\\pm${np.std(ok):.{prec}f}"


# ---------------------------------------------------------------- 回帰系共通
def envelope_stats(dataset, prefixes, key="train_eval_loss", fracs=(0.8, 0.95),
                   baseline="adam", progress_family=None):
    """{family: {(frac, basis): [per-seed ratio], "floor": [per-seed best]}}

    progress_family: マイルストーン目標 (loss0→lstar) をこの家族の
    エンベロープ到達進捗で定義する。None なら全セル共通の最良 loss。
    アブレーション表では "adam" を指定する — 崩壊バリアントが lstar を
    深掘りできず adam 自身が f=0.95 に未到達になると全比が未定義になる
    (初版 PDF の Table II が全行 --- になった原因) ため。"""
    data = collect(dataset, prefixes)
    seeds = [s for s in SEEDS if s in data]
    if not seeds:
        return None, []
    out = {}
    for s in seeds:
        hists = data[s]
        loss0 = max(h[key][0] for h in hists.values())
        if progress_family is None:
            lstar = min(min(h[key]) for h in hists.values())
        else:
            lstar = min(min(h[key]) for c, h in hists.items()
                        if c.split("@")[0] == progress_family)
        cells = sorted(hists.keys())
        fams = sorted({c.split("@")[0] for c in cells})

        def reach(c, f, basis):
            h = hists[c]
            xs = h.get(basis)
            if xs is None:
                xs = h["steps"] if not c.startswith("lbfgs") else None
            if xs is None:
                return None
            tgt = loss0 - f * (loss0 - lstar)
            return x_to(xs, h[key], tgt)

        def env(fam, f, basis):
            vals = [reach(c, f, basis) for c in cells
                    if c.split("@")[0] == fam]
            vals = [v for v in vals if v is not None]
            return min(vals) if vals else None

        for fam in fams:
            d = out.setdefault(fam, {})
            for f in fracs:
                for basis in ("steps", "time", "grad_evals"):
                    a = env(baseline, f, basis)
                    e = env(fam, f, basis)
                    d.setdefault((f, basis), []).append(
                        a / e if (a and e) else None)
            d.setdefault("floor", []).append(
                min(min(hists[c][key]) for c in cells
                    if c.split("@")[0] == fam))
    return out, seeds


def emit_regression():
    fams = ["adam", "adamw", "adam-cos", "adabelief", "adahessian",
            "bb", "bb-stab", "lbfgs", "rprop", "eagle", "eagle3",
            "eagle-dqn-cd", "kestrel-cos"]
    agg = {}  # (fam, frac, basis) -> [per-ds mean]
    for ds, tok in DS_TOK.items():
        # protoe2 = 旧 EAGLE / protoe3 = kestrel-cos (別プレフィックスで
        # 追加走行し、ここでマージ — protoe の上書き破壊を防ぐ)
        stats, seeds = envelope_stats(
            ds, ("protoe", "protoeh", "protoe2", "protoe3", "protoe5"))
        for fam in fams:
            ft = FAM_TOK[fam]
            d = (stats or {}).get(fam)
            for f in (0.8, 0.95):
                for basis, btok in (("steps", ""), ("time", "Time"),
                                    ("grad_evals", "Gev")):
                    name = f"R{tok}{ft}{FRAC_TOK[f]}{btok}"
                    if d is None:
                        emit(name, TBD)
                    else:
                        vals = d[(f, basis)]
                        emit(name, fmt_ratio(vals, len(seeds)))
                        ok = [v for v in vals if v is not None]
                        if ok and len(ok) == len(seeds):
                            agg.setdefault((fam, f, basis),
                                           []).append(np.mean(ok))
            emit(f"R{tok}{ft}Floor",
                 TBD if d is None else fmt_val(d["floor"]))
    # 概況レンジ (アブスト/イントロ用): eagle-dqn-cd の 3ds レンジ
    for f, ftok in ((0.8, "Mid"), (0.95, "Final")):
        for basis, btok in (("steps", ""), ("time", "Time")):
            v = agg.get(("eagle-dqn-cd", f, basis))
            if v:
                emit(f"RegEagle{ftok}{btok}Lo", f"{min(v):.2f}")
                emit(f"RegEagle{ftok}{btok}Hi", f"{max(v):.2f}")
            else:
                emit(f"RegEagle{ftok}{btok}Lo", TBD)
                emit(f"RegEagle{ftok}{btok}Hi", TBD)


def emit_sensitivity():
    """K / β_h 感度走査 (protoe4、既定 = protoe の eagle-dqn-cd)。"""
    # 既定行 (Table III) は Table I の R*EagleMid と同値だが、表ごとに
    # 独立した太字化のため専用名で複製する
    for tok in DS_TOK.values():
        emit(f"S{tok}DefaultMid", macros[f"R{tok}EagleMid"])
    variants = {"eagle-dqn-cd-k5": "Kfive", "eagle-dqn-cd-k80": "Keighty",
                "eagle-dqn-cd-bh05": "Bhlow", "eagle-dqn-cd-bh095": "Bhhigh"}
    for ds, tok in DS_TOK.items():
        stats, seeds = envelope_stats(ds, ("protoe", "protoe4"))
        for fam, vt in variants.items():
            d = (stats or {}).get(fam)
            if d is None:
                emit(f"S{tok}{vt}Mid", TBD)
                emit(f"S{tok}{vt}Floor", TBD)
            else:
                emit(f"S{tok}{vt}Mid",
                     fmt_ratio(d[(0.8, "steps")], len(seeds)))
                emit(f"S{tok}{vt}Floor", fmt_val(d["floor"]))


def emit_ablation():
    fams = ["eagle-orig", "eagle-dqn", "eagle3-noins", "eagle-dqn-cd",
            "eagle3", "eagle4-aj", "eagle4-noins"]
    for ds, tok in DS_TOK.items():
        # adam 基準は protoe (Stage A、同一プロトコル) から取る —
        # protoa/b/d のアブレーショングリッドは eagle 変種のみで走った
        stats, seeds = envelope_stats(
            ds, ("protoa", "protob", "protod", "protoe"),
            fracs=(0.8, 0.95), progress_family="adam")
        for fam in fams:
            ft = FAM_TOK[fam]
            d = (stats or {}).get(fam)
            name = f"A{tok}{ft}Final"
            if d is None:
                emit(name, TBD)
                emit(f"A{tok}{ft}Floor", TBD)
            else:
                emit(name, fmt_ratio(d[(0.95, "steps")], len(seeds)))
                emit(f"A{tok}{ft}Floor", fmt_val(d["floor"]))
        # 床インフレ (保護なし変種の床 / adam 床) の最大値 — 本文の
        # 「ベンチなしは床も劣化」の定量値
        if stats and "adam" in stats:
            adam_floor = np.mean(stats["adam"]["floor"])
            for fam in ("eagle-orig", "eagle-dqn", "eagle3-noins"):
                if fam in stats:
                    _floor_infl.append(
                        np.mean(stats[fam]["floor"]) / adam_floor)


_floor_infl = []


# ------------------------------------------------------------------ INR 系
def collect_inr(prefixes, image):
    """複数 prefix をマージ ({seed: {"opt@lr": hist}})。pilot_inr は
    (prefix, image, seed) 単位で metrics.json を上書きするため、追加変種は
    別 prefix で走らせてここで合流させる。"""
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    out = {}
    for prefix in prefixes:
        for path in sorted(RESULTS.glob(f"{prefix}_{image}_s*/metrics.json")):
            seed = int(path.parent.name.rsplit("_s", 1)[1])
            with open(path) as f:
                out.setdefault(seed, {}).update(json.load(f)["histories"])
    return out


def inr_env_reach(hists, fam, thr, basis):
    vals = []
    for k, h in hists.items():
        if k.split("@")[0] != fam:
            continue
        v = x_to(h[basis], [-p for p in h["psnr"]], -thr)
        if v is not None:
            vals.append(v)
    return min(vals) if vals else None


def inr_env_best(hists, fam):
    vals = [max(h["psnr"]) for k, h in hists.items()
            if k.split("@")[0] == fam]
    return max(vals) if vals else None


def emit_inr():
    fams = ["adamw", "adam-cos", "adabelief", "adahessian",
            "bb", "bb-stab", "lbfgs", "rprop", "eagle3", "eagle-dqn-cd",
            "kestrel-cos"]
    for image, tok in (("camera", "Cam"), ("astronaut", "Ast")):
        data = collect_inr(("inrv3", "inrv3c", "inrv3r"), image)
        seeds = sorted(data)
        for fam in fams:
            ft = FAM_TOK[fam]
            for thr, ttok in ((30, "Mid"), (32, "High")):
                for basis, btok in (("steps", ""), ("time", "Time")):
                    name = f"I{tok}{ft}{ttok}{btok}"
                    if not seeds:
                        emit(name, TBD)
                        continue
                    sp = []
                    for s in seeds:
                        a = inr_env_reach(data[s], "adam", thr, basis)
                        e = inr_env_reach(data[s], fam, thr, basis)
                        sp.append(a / e if (a and e) else None)
                    emit(name, fmt_ratio(sp, len(seeds)))
            name = f"I{tok}{ft}Final"
            if not seeds:
                emit(name, TBD)
            else:
                emit(name, fmt_val([inr_env_best(data[s], fam)
                                    for s in seeds], prec=2))
        name = f"I{tok}AdamFinal"
        if not seeds:
            emit(name, TBD)
        else:
            emit(name, fmt_val([inr_env_best(data[s], "adam")
                                for s in seeds], prec=2))


def emit_kodak():
    from scipy import stats as sps
    fams = ["adamw", "adam-cos", "adabelief", "adahessian",
            "bb-stab", "rprop", "eagle3", "eagle-dqn-cd", "kestrel-cos"]
    images = [f"kodim{i:02d}" for i in range(1, 25)]
    per_img = {}
    for img in images:
        d = collect_inr(("kodakb", "kodakc", "kodakr"), img)
        if 42 in d:
            per_img[img] = d[42]
    n = len(per_img)
    emit("KNumImages", str(n) if n else TBD)
    for fam in fams:
        ft = FAM_TOK[fam]
        if not per_img:
            for suf in ("Delta", "Wins", "PVal", "ReachMid", "ReachMidTime"):
                emit(f"K{ft}{suf}", TBD)
            continue
        deltas, reach_s, reach_t = [], [], []
        for img, hists in per_img.items():
            a = inr_env_best(hists, "adam")
            e = inr_env_best(hists, fam)
            if a is not None and e is not None:
                deltas.append(e - a)
            ar = inr_env_reach(hists, "adam", 30, "steps")
            er = inr_env_reach(hists, fam, 30, "steps")
            if ar and er:
                reach_s.append(ar / er)
            ar = inr_env_reach(hists, "adam", 30, "time")
            er = inr_env_reach(hists, fam, 30, "time")
            if ar and er:
                reach_t.append(ar / er)
        if deltas:
            m, s = np.mean(deltas), np.std(deltas)
            emit(f"K{ft}Delta", f"${m:+.2f}\\pm{s:.2f}$")
            emit(f"K{ft}DeltaMean", f"${m:+.2f}$")
            emit(f"K{ft}Wins", f"{sum(d > 0 for d in deltas)}/{len(deltas)}")
            # t 検定は n>=2 かつ有限 p のときのみ (部分データでの nan 対策)
            p = sps.ttest_1samp(deltas, 0.0)[1] if len(deltas) >= 2 else None
            if p is not None and np.isfinite(p):
                if p >= 1e-3:
                    emit(f"K{ft}PVal", f"{p:.2f}" if p >= 0.01
                         else f"{p:.3f}")
                else:
                    mant, ex = f"{p:.1e}".split("e")
                    emit(f"K{ft}PVal", f"{mant}\\times10^{{{int(ex)}}}")
            else:
                emit(f"K{ft}PVal", TBD)
        else:
            emit(f"K{ft}Delta", TBD)
            emit(f"K{ft}Wins", TBD)
            emit(f"K{ft}PVal", TBD)
        emit(f"K{ft}ReachMid",
             f"{np.median(reach_s):.2f}" if reach_s else TBD)
        emit(f"K{ft}ReachMidTime",
             f"{np.median(reach_t):.2f}" if reach_t else TBD)

    # kestrel-cos の直接対決 (paired、旗艦昇格判断の根拠):
    # vs adam-cos = 最終品質の統計的同等性 / vs KESTREL = 有意な上位互換
    for other, otok in (("adam-cos", "Acos"), ("eagle-dqn-cd", "K")):
        diffs = []
        for img, hists in per_img.items():
            a = inr_env_best(hists, "kestrel-cos")
            b = inr_env_best(hists, other)
            if a is not None and b is not None:
                diffs.append(a - b)
        if len(diffs) >= 2:
            p = sps.ttest_1samp(diffs, 0.0)[1]
            emit(f"KKcosVs{otok}Delta",
                 f"${np.mean(diffs):+.2f}\\pm{np.std(diffs):.2f}$")
            emit(f"KKcosVs{otok}PVal", f"{p:.2g}" if p >= 1e-3 else
                 "{}\\times10^{{{}}}".format(*[
                     v if i == 0 else int(v)
                     for i, v in enumerate(f"{p:.1e}".split("e"))]))
        else:
            emit(f"KKcosVs{otok}Delta", TBD)
            emit(f"KKcosVs{otok}PVal", TBD)


def emit_diagnostics():
    """2 次関数診断 (experiments/diagnose_secant.py) のサマリー。"""
    path = RESULTS / "analysis" / "secant_diagnosis.json"
    if not path.exists():
        for n in ("DiagContractionSep", "DiagContractionHalf",
                  "DiagContractionFull"):
            emit(n, TBD)
        return
    with open(path) as f:
        d = json.load(f)
    emit("DiagContractionSep",
         f"{d['ideal_contraction_median']:.3f}")
    emit("DiagContractionHalf",
         f"{d['coupled_contraction_median']['0.5']:.3f}")
    emit("DiagContractionFull",
         f"{d['coupled_contraction_median']['1.0']:.3f}")


def emit_stepcost_bench():
    """N3: adam(foreach)/adam(fused)/KESTREL の per-step マイクロベンチ。"""
    path = RESULTS / "analysis" / "stepcost_bench.json"
    names = {"adam_foreach": "Adam", "adam_fused": "AdamFused",
             "kestrel_fused": "Kestrel"}
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        for scale, stok in (("regression", "Reg"), ("siren", "Siren")):
            for k, ktok in names.items():
                emit(f"Bench{stok}{ktok}", f"{d[scale][k]:.2f}")
    else:
        for stok in ("Reg", "Siren"):
            for ktok in names.values():
                emit(f"Bench{stok}{ktok}", TBD)


def emit_stepcost():
    """W2: protoe 実測の per-step 時間比 (stock Adam / fused KESTREL)。"""
    ratios = []
    for ds in DS_TOK:
        agg = {}
        for path in RESULTS.glob(f"protoe_{ds}_lr*_s*/metrics.json"):
            with open(path) as f:
                h = json.load(f)["histories"]
            for name, hh in h.items():
                t, st = hh.get("time"), hh.get("steps")
                if t and st and len(t) > 3 and st[-1] > st[1]:
                    agg.setdefault(name.split("@")[0] if "@" in name else name,
                                   []).append((t[-1]-t[1])/(st[-1]-st[1]))
        if "adam" in agg and "eagle-dqn-cd" in agg:
            ratios.append(np.mean(agg["adam"]) / np.mean(agg["eagle-dqn-cd"]))
    if ratios:
        emit("StepRatioLo", f"{min(ratios):.2f}")
        emit("StepRatioHi", f"{max(ratios):.2f}")
    else:
        emit("StepRatioLo", TBD)
        emit("StepRatioHi", TBD)


def emit_diag_inr():
    """W3/W4: INR 診断 (diagnose_kestrel_inr.py) + 回帰ジャンプ率実測。"""
    path = RESULTS / "analysis" / "kestrel_inr_diag.json"
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        bins = np.array(d["bins"])
        mid = np.sqrt(bins[:-1] * bins[1:])
        for img, tok in (("camera", "Cam"), ("astronaut", "Ast")):
            r = d[img]
            emit(f"DiagJump{tok}", f"{np.mean(r['jump_frac'])*100:.0f}")
            emit(f"DiagBench{tok}", f"{np.mean(r['bench_frac'])*100:.0f}")
            h = np.array(r["hist"])
            c = h.cumsum() / h.sum()
            p999 = mid[np.searchsorted(c, 0.999)]
            mant, ex = f"{p999:.0e}".split("e")
            emit(f"DiagMagPnnn{tok}", f"{mant}\\times10^{{{int(ex)}}}")
    else:
        for tok in ("Cam", "Ast"):
            for n in (f"DiagJump{tok}", f"DiagBench{tok}",
                      f"DiagMagPnnn{tok}"):
                emit(n, TBD)
    # 回帰スイートの累積ジャンプ率 (protoe の eagle_ratio)
    vals = []
    for path in RESULTS.glob("protoe_*_s*/metrics.json"):
        with open(path) as f:
            h = json.load(f)["histories"]
        hh = h.get("eagle-dqn-cd")
        if hh and hh.get("eagle_ratio"):
            er = [x for x in hh["eagle_ratio"] if x is not None]
            if er:
                vals.append(er[-1] * 100)
    if vals:
        emit("RegJumpLo", f"{min(vals):.0f}")
        emit("RegJumpHi", f"{max(vals):.0f}")
    else:
        emit("RegJumpLo", TBD)
        emit("RegJumpHi", TBD)


def emit_constants():
    # SIREN 2-256x3-C: 論文で引くパラメータ数 (グレースケール C=1)
    n = (2 * 256 + 256) + 2 * (256 * 256 + 256) + (256 * 1 + 1)
    emit("SirenParams", f"{n / 1000:.0f}k")
    emit("CooldownK", "20")
    emit("BetaH", "0.8")


def parse_lead(value):
    """マクロ値の先頭数値 (表示値) を取り出す。--- や TBD は None。"""
    import re as _re
    if "TBD" in value or value.strip() == "---":
        return None
    m = _re.search(r"[-+]?\d+\.?\d*", value)
    return float(m.group(0)) if m else None


def apply_bold():
    """各表の列ごとに最高性能セルを \textbf 化する (表示値で比較、
    同値タイは全て太字)。p 値列と別基準の注釈行は対象外。"""
    t1 = ["Adam", "Adamw", "Adamcos", "Rprop", "Adabelief", "Adahessian",
          "Bb", "Bbstab", "Eaglearxiv", "Lbfgs", "Eagle"]
    t2 = ["Eagleorig", "Eagledqn", "Eagleswitchonly", "Eagle", "Eaglethree",
          "Eagleaj", "Eaglenoins"]
    t4 = ["Adamw", "Adamcos", "Rprop", "Adabelief", "Adahessian", "Bbstab",
          "Lbfgs", "Eagle", "Kestrelcos"]
    t5 = ["Adamw", "Adamcos", "Rprop", "Adabelief", "Adahessian", "Bbstab",
          "Eagle", "Kestrelcos"]
    cols = []
    for tok in DS_TOK.values():
        cols += [([f"R{tok}{f}Mid" for f in t1], True),
                 ([f"R{tok}{f}Final" for f in t1], True),
                 ([f"R{tok}{f}Floor" for f in t1], False),
                 ([f"A{tok}{f}Final" for f in t2], True),
                 ([f"S{tok}{v}Mid" for v in
                   ("Kfive", "Keighty", "Bhlow", "Bhhigh")]
                  + [f"S{tok}DefaultMid"], True)]
    for itok in ("Cam", "Ast"):
        cols += [([f"I{itok}{f}Mid" for f in t4], True),
                 ([f"I{itok}{f}Final" for f in t4]
                  + [f"I{itok}AdamFinal"], True)]
    cols += [([f"K{f}Delta" for f in t5], True),
             ([f"K{f}Wins" for f in t5], True),
             ([f"K{f}ReachMid" for f in t5], True)]
    for names, higher in cols:
        vals = {n: parse_lead(macros[n]) for n in names if n in macros}
        ok = {n: v for n, v in vals.items() if v is not None}
        if not ok:
            continue
        best = max(ok.values()) if higher else min(ok.values())
        for n, v in ok.items():
            if abs(v - best) < 1e-9:
                macros[n] = "\\textbf{" + macros[n] + "}"


def main():
    emit_constants()
    emit_diagnostics()
    emit_stepcost()
    emit_stepcost_bench()
    emit_diag_inr()
    emit_regression()
    emit_sensitivity()
    emit_ablation()
    emit("AblFloorInflMax",
         f"{max(_floor_infl):.1f}" if _floor_infl else TBD)
    emit_inr()
    emit_kodak()
    apply_bold()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("% AUTO-GENERATED by ictai/gen_macros.py -- DO NOT EDIT.\n"
                "% Regenerate: python ictai/gen_macros.py\n")
        for name, value in macros.items():
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")
    n_tbd = sum(1 for v in macros.values() if v == TBD)
    print(f"{len(macros)} macros -> {OUT} (TBD: {n_tbd})")


if __name__ == "__main__":
    main()
