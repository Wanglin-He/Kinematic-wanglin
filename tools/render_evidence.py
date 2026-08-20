"""Four orientation views per record, for the human review page.

Not evidence for any verdict. The scoring pass has already run and frozen its results;
this re-poses each asset at its reference configuration and takes four pictures from the
same four angles, so a person can see which object they are being asked about. Every
number on the page comes from the report, not from these.
"""
import json, pathlib, sys
import numpy as np
from evo_p0p3.p0.loader import load_contract
from evo_p0p3.p3 import gate, review
from evo_p0p3.p3.sweep import Sweeper

C = "/Users/wanglin/Desktop/articraft-data/cache/record_materialization"
JOBS = [
 ("棱柱 · 空气炸锅","air_fryer","rec_air_fryer_with_pullout_basket_14ac4c8c9e35416b864382571fa27e62"),
 ("旋转 · 摄像机翻转屏","camcorder","rec_camcorder_with_flipout_screen_b5753ec789b546b0a058ca52bbcd590d"),
 ("旋转 · 笔记本翻盖","laptop_clamshell","rec_laptop_clamshell_d201e66211d4426096f1625b7b4fc1cf"),
 ("旋转 · 手套箱门","glove_compartment_door","rec_glove_compartment_door_927c892914044efcb1cb608ac19db1b7"),
 ("旋转 · 吊扇灯盘","ceiling_fan","rec_ceiling_fan_a743cb8b7b834f689d8e18995b53306a"),
 ("倾斜 · 钻床斜台","drillpress_tilt_table","rec_drillpress_tilt_table_0001"),
 ("转子 · 离心榨汁机","centrifugal_juicer","rec_centrifugal_juicer_with_articulated_components_8cf981d5e9bd4079bfcaa4a2e5ace079"),
 ("转子 · 曲柄脚踏","bicycle_crankset","rec_bicycle_crankset_and_pedal_assembly_815cd8c51ee342719a051869fe2b7325"),
 ("调节 · 健身车","stationary_exercise_bike","rec_stationary_exercise_bike_541413c0121143a2bed6faede26e1fdc"),
 ("调节 · 灶台","stove_top","rec_stove_top_0002"),
]

out = []
for zh, name, rid in JOBS:
    c = load_contract(f"contracts/pilot/{name}.yaml")
    ad = gate.admit(f"{C}/{rid}/model.urdf", c,
                    binding_table=f"contracts/pilot/bindings/{name}.yaml", diagnostic=True)
    if ad.binding is None:
        print("skip", name, file=sys.stderr); continue
    rep = json.loads(pathlib.Path(f"out/pilot/{rid}.json").read_text())
    ref = np.asarray(Sweeper(c, ad.binding).schedule().samples[0].qpos, float)
    views = review.surround(ad.asset, ad.binding, ref)

    fails = [{"predicate": x["predicate"], "subject": x["subject"], "reason": x.get("reason",""),
              "measured": x.get("measured") or {}, "threshold": x.get("threshold") or {},
              "evidence": x.get("evidence") or {}}
             for x in rep["claims"] if x["verdict"] == "fail"]
    prompt = ""
    pj = pathlib.Path(f"{C}/{rid}/prompt.txt")
    if pj.exists(): prompt = pj.read_text(encoding="utf-8").strip()
    out.append({"zh": zh, "name": name, "rid": rid, "profile": rep["profile"],
                "n_claims": len(rep["claims"]),
                "n_na": sum(1 for x in rep["claims"] if x["verdict"] == "na"),
                "prompt": prompt, "views": list(views), "fails": fails})
    print(f"{zh}: 4 张视图 · {len(fails)} 条判负", file=sys.stderr)

pathlib.Path("/tmp/shots.json").write_text(json.dumps(out, ensure_ascii=False))
kb = pathlib.Path("/tmp/shots.json").stat().st_size // 1024
print(f"\n{len(out)} 个资产 · {len(out)*4} 张图 · {kb} KB", file=sys.stderr)
