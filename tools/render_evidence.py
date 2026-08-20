import json, re, pathlib, sys
import numpy as np, mujoco
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

def parts_of(subject, ev):
    if ev.get("worst_pair"): return tuple(ev["worst_pair"])
    m = re.match(r"^(.+?)\+(.+?)@", subject)
    if m: return (m.group(1), m.group(2))
    if ev.get("part"): return (ev["part"],)
    return ()

out = []
for zh, name, rid in JOBS:
    c = load_contract(f"contracts/pilot/{name}.yaml")
    ad = gate.admit(f"{C}/{rid}/model.urdf", c,
                    binding_table=f"contracts/pilot/bindings/{name}.yaml", diagnostic=True)
    if ad.binding is None:
        print("skip", name, file=sys.stderr); continue
    asset, b = ad.asset, ad.binding
    m = asset.model
    rep = json.loads(pathlib.Path(f"out/pilot/{rid}.json").read_text())
    sched = Sweeper(c, b).schedule()
    by_label = {s.label: np.asarray(s.qpos, float) for s in sched.samples}
    ref = by_label["reference"]

    shots = []
    for cl in rep["claims"]:
        if cl["verdict"] != "fail": continue
        pred, subj = cl["predicate"], cl["subject"]
        ev, meas = cl.get("evidence") or {}, cl.get("measured") or {}
        parts = parts_of(subj, ev)
        if not parts:   # KF1 的证据里没有 part 字段，从契约的关节声明反查
            for j in c.kinematic_claims.joints:
                if j.id == subj:
                    parts = (j.part,) + ((j.parent,) if j.parent in b.parts else ())
                    break
        parts = tuple(p for p in parts if p in b.parts)
        if not parts: continue
        joints = False; cap = ""; fail = None

        if pred.startswith("KF3"):
            lab = ev.get("worst_configuration") or ev.get("closest_configuration")
            fail = by_label.get(lab)
            cap = f"构型 {lab}"
            if fail is None: fail = ref; cap = f"构型 {lab}（未在计划中，退回参考姿态）"
        else:
            # KF1：把该关节推到模型自己的极限，让「它实际能动到哪」看得见
            jid = None
            for j in c.kinematic_claims.joints:
                if j.id == subj: jid = j; break
            fail = ref.copy()
            if jid is not None and jid.part in b.parts:
                bd = b.root_body(jid.part)
                if int(m.body_jntnum[bd]) >= 1:
                    k = int(m.body_jntadr[bd]); adr = int(m.jnt_qposadr[k])
                    lo, hi = (float(m.jnt_range[k][0]), float(m.jnt_range[k][1])) \
                             if int(m.jnt_limited[k]) else (0.0, 1.0)
                    fail[adr] = hi if abs(hi - ref[adr]) > abs(lo - ref[adr]) else lo
                    unit = jid.range.unit if jid.range else ""
                    if pred == "KF1.range_and_reference":
                        cap = (f"模型自己只能到 {fail[adr]:+.4g} {unit}"
                               f"（契约声明 {meas.get('model_span','?')} vs "
                               f"{cl.get('threshold',{}).get('declared_span','?')}）")
                    else:
                        cap = f"该关节推到模型自己的极限 {fail[adr]:+.4g} {unit}"
            joints = pred in ("KF1.anchor", "KF1.axis_semantic")
            if joints: cap += "（细线为 MuJoCo 画出的关节轴）"
            if pred == "KF1.parent" and ev.get("nearest_declared_ancestor") in b.parts:
                parts = (parts[0], ev["nearest_declared_ancestor"])

        sh = review.shoot(asset, b, ref, fail, parts, caption=cap, show_joints=joints)
        shots.append({"predicate": pred, "subject": subj, "measured": meas,
                      "threshold": cl.get("threshold") or {}, "evidence": ev,
                      "why": cl.get("message") or cl.get("reason") or "",
                      "parts": list(parts), "caption": cap,
                      "ref": sh.reference_png, "fail": sh.failing_png})
    prompt = ""
    pj = pathlib.Path(f"{C}/{rid}/prompt.txt")
    if pj.exists(): prompt = pj.read_text(encoding="utf-8").strip()
    out.append({"zh": zh, "name": name, "rid": rid, "profile": rep["profile"],
                "n_claims": len(rep["claims"]),
                "n_na": sum(1 for x in rep["claims"] if x["verdict"]=="na"),
                "prompt": prompt, "shots": shots})
    print(f"{zh}: {len(shots)} 条判负已渲染", file=sys.stderr)

pathlib.Path("/tmp/shots.json").write_text(json.dumps(out, ensure_ascii=False))
kb = pathlib.Path("/tmp/shots.json").stat().st_size//1024
print(f"\n共 {sum(len(a['shots']) for a in out)} 条判负 · {sum(len(a['shots']) for a in out)*2} 张图 · {kb} KB", file=sys.stderr)
