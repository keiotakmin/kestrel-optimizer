"""DIV2K validation から confirmatory 用の 16 枚を用意する (継続研究 A)。

Kodak-24 は ICTAI 版までの探索的分析で既に使い尽くしているため、
Kodak 内の tuning/evaluation split は selection bias を減らすだけで
exploratory 分析から独立ではない。lock 済み設定を**一度も見ていない**
データで一回だけ評価するために、DIV2K validation を使う。

事前登録する規則 (結果を一切見る前に固定する):
  - DIV2K_valid_HR (0801-0900) の **ファイル名昇順で先頭 16 枚** = 0801-0816
  - Kodak とプロトコルを完全一致させるため 768x512 (縦長は 512x768) に
    bicubic で縮小する。解像度を変えると lock 済み lr が転移しないため。
  - 以後この 16 枚に対して設定を選び直さない。走行は 1 回だけ。

ライセンス: DIV2K (Agustsson & Timofte, NTIRE 2017) は学術研究用途で提供。
再配布はせず、取得スクリプトと URL のみを artifact に含める。

実行: python experiments/prepare_div2k.py
"""

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "div2k_raw" / "DIV2K_valid_HR.zip"
URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
DST = ROOT / "data" / "div2k16"
REG = Path(__file__).with_name("div2k_split.json")
N_IMAGES = 16
LONG, SHORT = 768, 512


def fetch():
    """SRC が無ければ取得する。再配布はせず取得手順だけを持つ。"""
    if SRC.exists():
        return
    SRC.parent.mkdir(parents=True, exist_ok=True)
    tmp = SRC.with_suffix(".zip.part")
    print(f"downloading {URL} -> {SRC} (about 430 MB)")
    urllib.request.urlretrieve(URL, tmp)
    tmp.rename(SRC)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    fetch()
    z = zipfile.ZipFile(SRC)
    names = sorted(x for x in z.namelist() if x.lower().endswith(".png"))
    picked = names[:N_IMAGES]

    out_names = []
    for src in picked:
        stem = Path(src).stem
        tag = f"div2k{stem}"
        out = DST / f"{tag}.png"
        out_names.append(tag)
        if out.exists():
            continue
        with z.open(src) as f:
            img = Image.open(f).convert("RGB")
        w, h = img.size
        size = (LONG, SHORT) if w >= h else (SHORT, LONG)
        img.resize(size, Image.BICUBIC).save(out)
        print(f"  {stem}: {w}x{h} -> {size[0]}x{size[1]}")

    reg = {
        "_registered": "2026-08-29",
        "_purpose": "IEEE Access 拡張版の confirmatory 評価 (継続研究 A)",
        "_rule": ("DIV2K_valid_HR のファイル名昇順で先頭 16 枚。"
                  "結果を見る前に固定した。以後選び直さない。"),
        "_resize": (f"{LONG}x{SHORT} (縦長は {SHORT}x{LONG}) bicubic。"
                    "Kodak とプロトコルを一致させるため。"),
        "_license": ("DIV2K (NTIRE 2017, Agustsson & Timofte) 学術研究用途。"
                     "再配布しない。取得元 = "
                     "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
                     "DIV2K_valid_HR.zip"),
        "_lr_source": "experiments/kodak_lr_lock.json (Kodak tuning で lock)",
        "images": out_names,
    }
    # 事前登録ファイルは write-once。すでに存在する場合は画像リストの一致だけ
    # 確認して書き換えない (登録内容を後から差し替えないため)。
    if REG.exists():
        old = json.loads(REG.read_text())
        if old.get("images") != out_names:
            raise SystemExit(
                f"{REG} は既に登録済みで、画像リストが今回の生成結果と異なる。"
                "登録は後から変更しない規則なので、意図的な再登録なら"
                "ファイルを手で退避してから実行すること。")
        print(f"\n{len(out_names)} 枚 -> {DST}\n登録は既存を維持 -> {REG}")
    else:
        REG.write_text(json.dumps(reg, ensure_ascii=False, indent=2))
        print(f"\n{len(out_names)} 枚 -> {DST}\n登録 -> {REG}")

    arr = np.asarray(Image.open(DST / f"{out_names[0]}.png"))
    print(f"確認: {out_names[0]} shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
