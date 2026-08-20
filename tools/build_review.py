import json, pathlib, html, textwrap
D = json.loads(pathlib.Path("/tmp/shots.json").read_text())
E = html.escape

def cn(f, parts_hint=None):
    """One sentence saying what is wrong, built from the numbers the predicate recorded."""
    p, m, t, ev = f["predicate"], f["measured"], f["threshold"], f["evidence"]
    g = lambda d, k, dv=None: d.get(k, dv)
    pair = ev.get("worst_pair") or []
    subj = f["subject"]
    if not pair and "+" in subj and "@" in subj:
        pair = subj.split("@")[0].split("+")
    a = pair[0] if pair else ev.get("part", "?")
    b = pair[1] if len(pair) > 1 else "?"
    try:
        if p == "KF1.parent":
            return (f"{ev.get('part','该部件')} 实际挂在 <b>{g(m,'nearest_declared_ancestor')}</b> 下面"
                    f"（上溯 {g(m,'links_up')} 层），不是契约声明的父件——它跟着动的不是该跟的那个。")
        if p == "KF1.axis_semantic":
            return (f"这根轴与声明的方向差 <b>{g(m,'deviation_deg')}°</b>，超过允许的 {g(t,'axis_angle_deg')}°。")
        if p == "KF1.anchor" and "edge_inset" in m:
            return (f"轴线扎进 {ev.get('part','部件')} 内部 <b>{g(m,'edge_inset')*100:.1f}%</b>"
                    f"（跨度 {g(m,'extent_m')} m），超过允许的 {g(t,'anchor_edge_inset_max')*100:.0f}%"
                    f"——它绕的是<b>中间</b>不是边缘。")
        if p == "KF1.anchor":
            return (f"轴线偏离 {ev.get('part','部件')} 中心 <b>{g(m,'offset_over_diagonal')*100:.1f}%</b>"
                    f"（{g(m,'offset_m')} m ÷ 自身对角线 {g(m,'part_diagonal_m')} m），"
                    f"超过允许的 {g(t,'anchor_center_offset_max')*100:.0f}%。")
        if p == "KF1.range_and_reference":
            ms, ds = g(m,'model_span'), g(t,'declared_span')
            return (f"模型只能动 <b>{ms}</b>，契约声明 <b>{ds}</b>，相差 {abs(ms-ds)/ds*100:.0f}%。")
        if p == "KF3.forbidden_pair":
            return (f"{a} 与 {b} 最深互相嵌入 <b>{g(m,'max_penetration_m')} m</b>"
                    f"（允许 {g(t,'forbidden_penetration_m')} m），最坏构型 "
                    f"<code>{ev.get('worst_configuration')}</code>，共查 {g(m,'samples_evaluated')} 个构型。")
        if p == "KF3.state_reachability":
            return (f"状态 <b>{subj}</b> 停不干净：{a} 与 {b} 嵌入 <b>{g(m,'max_penetration_m')} m</b>"
                    f"（允许 {g(t,'forbidden_penetration_m')} m），该状态共 {g(m,'configurations_in_state')} 个构型。")
        if p == "KF3.required_contact":
            return (f"{a} 与 {b} 全程最近只到 <b>{g(m,'min_clearance_m')} m</b>，要求 ≤ {g(t,'required_contact_m')} m"
                    f"——契约说这里该有接触，<b>但它们从来没碰上</b>。")
        if p == "KF2.bound":
            return "模型里没有任何约束强制这条耦合，两个关节各动各的。"
        if p == "KF2.expected_dof":
            return f"剩余自由度 {g(m,'remaining_dof')}，契约声明 {g(t,'expected_dof')}。"
    except Exception:
        pass
    return E(f.get("reason") or "")

CSS = """
:root{--ink:#1a2434;--muted:#66748a;--navy:#16324f;--teal:#0a7d76;--teal-s:#e4f4f0;
--paper:#f6f4ee;--card:#fffefb;--line:#dbd8d0;--red:#a03f2e;--red-s:#fbe9e4;
--green:#22715a;--green-s:#e7f4ee;--gold:#a06c17;--gold-s:#faf1db;
--mono:"SFMono-Regular",Consolas,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);line-height:1.62;background:var(--paper);
font-family:Inter,ui-sans-serif,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
code{font-family:var(--mono);font-size:.87em;padding:.05em .28em;border-radius:4px;
color:#075f5b;background:var(--teal-s)}
header.top{color:#fff;background:linear-gradient(118deg,#13304e,#0e6c66)}
.hero{max-width:1080px;margin:0 auto;padding:28px 24px 24px}
h1{margin:5px 0 0;font-family:Georgia,"Songti SC",serif;font-size:clamp(1.45rem,3vw,2.1rem);font-weight:600}
.eyebrow{color:#c9ece7;font-size:.71rem;font-weight:850;letter-spacing:.14em}
.sub{margin:8px 0 0;color:#d6e6ec;font-size:.92rem;max-width:720px}
.bar{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:10px;flex-wrap:wrap;
padding:8px 24px;border-bottom:1px solid var(--line);background:#fffefbee;backdrop-filter:blur(8px)}
.bar b{font-size:.85rem;color:var(--navy)}.bar .sp{flex:1}
button{padding:5px 12px;border:1px solid var(--navy);border-radius:7px;background:var(--navy);
color:#fff;font:inherit;font-size:.81rem;cursor:pointer}
button.g{background:#fff;color:var(--navy)}button:hover{opacity:.86}
#tally{font-family:var(--mono);font-size:.79rem;color:var(--muted)}
input[type=text]{padding:4px 8px;border:1px solid var(--line);border-radius:6px;font:inherit;font-size:.83rem}
main{max-width:1080px;margin:0 auto;padding:0 24px 56px}
h2{margin:22px 0 8px;color:var(--navy);font-family:Georgia,"Songti SC",serif;font-size:1.28rem}
p{margin:7px 0;font-size:.91rem}
.how{margin-top:18px;padding:16px 18px;border:1px solid var(--line);border-radius:12px;background:var(--card)}
.how h3{margin:15px 0 4px;font-size:.96rem;color:var(--navy)}
.how h3:first-of-type{margin-top:0}
.how ul{margin:6px 0;padding-left:1.1rem;font-size:.89rem}
.f{margin:7px 0;padding:9px 12px;border-radius:8px;background:var(--teal-s);color:#0b4744;
font-family:var(--mono);font-size:.775rem;line-height:1.62;white-space:pre-wrap;overflow-x:auto}
.n{margin:8px 0;padding:8px 12px;border-left:3px solid var(--gold);border-radius:6px;
background:var(--gold-s);color:#64450e;font-size:.85rem}
.asset{margin-top:18px;border:1px solid var(--line);border-radius:13px;background:var(--card);overflow:hidden}
.ah{padding:12px 18px 0}
.ah h3{margin:0;font-size:1.04rem;color:var(--navy)}
.ah h3 .idx{display:inline-block;margin-right:8px;padding:1px 8px;border-radius:6px;
background:var(--navy);color:#fff;font-family:var(--mono);font-size:.78rem;vertical-align:2px}
.rid{margin:3px 0 0;font-family:var(--mono);font-size:.72rem;color:var(--muted);word-break:break-all}
.prompt{margin:9px 0 0;padding:8px 11px;border-radius:7px;background:#f2efe7;color:#3f4f68;font-size:.83rem}
.views{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;margin-top:12px;background:var(--line)}
.views figure{margin:0;background:#000}
.views img{display:block;width:100%;height:auto}
.views figcaption{padding:4px 0;background:#f4f2ec;color:var(--muted);text-align:center;
font-family:var(--mono);font-size:.72rem}
.scores{display:flex;gap:7px;flex-wrap:wrap;padding:11px 18px 0}
.sc{padding:2px 10px;border:1px solid var(--line);border-radius:999px;background:#fff;
font-family:var(--mono);font-size:.79rem}
.sc.lo{color:var(--red);border-color:#dcaaa3;background:var(--red-s);font-weight:700}
.sc.hi{color:var(--green);border-color:#a8cebe;background:var(--green-s)}
.fails{padding:10px 18px 0}
.fails>b{font-size:.87rem;color:var(--navy)}
.fl{margin:7px 0;padding:9px 12px;border-left:3px solid var(--red);border-radius:7px;background:var(--red-s)}
.fl .h{font-family:var(--mono);font-size:.8rem;color:#8d3423;font-weight:700}
.fl .h em{font-style:normal;font-weight:400;color:#a86a5c;margin-left:6px}
.fl .t{margin:3px 0 0;color:#742d1e;font-size:.875rem;line-height:1.6}
.clean{margin:7px 0;padding:9px 12px;border-left:3px solid var(--green);border-radius:7px;
background:var(--green-s);color:#1c5443;font-size:.88rem}
.rev{margin:12px 18px 16px;padding:10px 13px;border:1px dashed #b9c4d2;border-radius:9px;background:#f7fafc}
.rev>b{font-size:.84rem;color:var(--navy)}
textarea{width:100%;min-height:64px;margin-top:6px;padding:7px 9px;border:1px solid var(--line);
border-radius:7px;font:inherit;font-size:.87rem;resize:vertical;background:#fff}
footer{margin-top:28px;padding:15px 24px;color:#cfe1e8;background:var(--navy);text-align:center;font-size:.8rem}
@media(max-width:700px){.views{grid-template-columns:repeat(2,1fr)}}
"""

JS = """
const KEY='p0p3-review-v2';
const load=()=>{try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return{}}};
const who=()=>document.getElementById('who').value.trim();
function collect(){
  const o={reviewer:who(),at:new Date().toISOString(),notes:[]};
  document.querySelectorAll('textarea[data-rid]').forEach(t=>{
    const v=t.value.trim();
    if(v)o.notes.push({record_id:t.dataset.rid,object:t.dataset.obj,note:v});
  });
  return o;
}
function persist(){
  const st={__who:who()};
  document.querySelectorAll('textarea[data-rid]').forEach(t=>st[t.dataset.rid]=t.value);
  localStorage.setItem(KEY,JSON.stringify(st));
  document.getElementById('tally').textContent=`已填 ${collect().notes.length}/${TOTAL} 个资产`;
}
function restore(){
  const st=load();
  if(st.__who)document.getElementById('who').value=st.__who;
  document.querySelectorAll('textarea[data-rid]').forEach(t=>{if(st[t.dataset.rid])t.value=st[t.dataset.rid]});
  persist();
}
function download(){
  const b=new Blob([JSON.stringify(collect(),null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download=`p3-review-${who()||'anon'}.json`;a.click();
}
function copyJSON(){
  navigator.clipboard.writeText(JSON.stringify(collect(),null,2)).then(()=>{
    const b=document.getElementById('cp'),o=b.textContent;b.textContent='已复制 ✓';
    setTimeout(()=>b.textContent=o,1400)});
}
document.addEventListener('DOMContentLoaded',()=>{restore();
  document.addEventListener('input',e=>{if(e.target.tagName==='TEXTAREA'||e.target.id==='who')persist()})});
"""

blocks = []
for i, a in enumerate(D, 1):
    p = a["profile"]
    def sc(k):
        v = p.get(k)
        if v is None: return f'<span class="sc">{k} —</span>'
        c = "hi" if v >= 0.999 else ("lo" if v < 0.75 else "")
        return f'<span class="sc {c}">{k} {v:.2f}</span>'
    views = "".join(
        f'<figure><img src="data:image/png;base64,{v}" alt=""><figcaption>{d}°</figcaption></figure>'
        for v, d in zip(a["views"], (0, 90, 180, 270)))
    if a["fails"]:
        items = "".join(
            f'<div class="fl"><div class="h">{E(f["predicate"])}<em>{E(f["subject"])}</em></div>'
            f'<p class="t">{cn(f)}</p></div>' for f in a["fails"])
        fails = f'<div class="fails"><b>判负 {len(a["fails"])} 条</b>{items}</div>'
    else:
        fails = ('<div class="fails"><b>判负</b>'
                 '<div class="clean">零条判负 —— 每一条可评估的声明都通过了。</div></div>')
    blocks.append(f"""
  <div class="asset">
    <div class="ah">
      <h3><span class="idx">{i:02d}</span>{E(a["zh"])}</h3>
      <p class="rid">{E(a["rid"])}</p>
      {'<p class="prompt"><b>Prompt：</b>' + E(a["prompt"][:700]) + '</p>' if a["prompt"] else ''}
    </div>
    <div class="views">{views}</div>
    <div class="scores">{sc("KF1")}{sc("KF2")}{sc("KF3")}
      <span class="sc">{a["n_claims"]} 条声明</span><span class="sc">{a["n_na"]} 条 N/A</span></div>
    {fails}
    <div class="rev"><b>人工审核</b>
      <textarea data-rid="{E(a["rid"])}" data-obj="{E(a["zh"])}"
        placeholder="看完 prompt 和四张图，写下你自己的判断：这个物体做得对不对、哪里有问题、问题有多严重。"></textarea>
    </div>
  </div>""")

total = len(D)
n_fail = sum(len(a["fails"]) for a in D)
page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><title>P3 运动学复核</title>
<style>{CSS}</style></head><body>
<header class="top"><div class="hero">
  <div class="eyebrow">EVO-ARTICRAFT · P3 · 运动学保真度</div>
  <h1>10 个资产 · {n_fail} 条判负 · 人工复核</h1>
  <p class="sub">先说清楚 KF1/KF2/KF3 是怎么判的，然后逐个资产列出它判负的每一条，
  每条都带上它是用哪个数、在哪个构型上决定的。</p>
</div></header>

<div class="bar"><b>审核人</b><input type="text" id="who" placeholder="你的名字" size="10">
  <span id="tally"></span><span class="sp"></span>
  <button class="g" id="cp" onclick="copyJSON()">复制 JSON</button>
  <button onclick="download()">导出 JSON</button></div>

<main>
<section class="how">
  <h2>判定方法</h2>
  <p>资产载入 MuJoCo，摆到指定构型，读世界坐标和几何距离。<b>全程位置层</b>
  （<code>mj_forward</code>，不步进、不涉及质量摩擦），<b>不调用任何模型</b>——
  只有查表、点积、计数、距离。页面上的渲染图在评分之后补的，<b>改不了任何分数</b>。</p>

  <h3>KF1 · 关节配置保真度 —— 9 条</h3>
  <p>一条关节声明说了七件事，逐件判；另加两条，因为有两类失败没有字段覆盖。</p>
  <div class="f">parent               A(b*) = 沿 body_parentid 上溯的第一个已绑定部件，须等于声明的父件
type                 jnt_type[k] = μ(声明)
dof_composition      body_jntnum[b*] = 1                        这个部件只挂一个关节
axis_semantic        c = |â·û|，û = −g/|g|
                     vertical: c ≥ cos15°   horizontal: c ≤ sin15°
axis_admits_motion   沿声明轴向推 δ = min(0.25·行程, 0.05) 后，与父件距离 d₁ ≥ −ε
anchor               on_edge_of:        轴线内嵌 = min(|min s|,|max s|)/(max s − min s) ≤ 0.15
                     through_center_of: |质心到轴线| / 部件自身对角线 ≤ 0.05
range_and_reference  |模型行程 − 声明行程| / 声明行程 ≤ 0.02
rigid_follower       随动件须在主动件的「不跨越任何关节可达集」内
travel_scale         部件自身包围盒对角线 / 行程 ≥ 0.5          仅 slide，抓「有关节没实体」</div>

  <h3>KF2 · 机械耦合保真度 —— 4 条</h3>
  <p>不看几何，看模型自己的<b>约束表</b>。MuJoCo 的 <code>eq_data</code> 写成
  <code>q₁ = a₀ + a₁q₂ + …</code>，与契约的<code>依赖 = 系数 × 独立 + 偏移</code>逐字段对应。</p>
  <div class="f">KF2 = mean_g  1[bound(g)] × 1[ max_q |残差| ≤ ε_g ]

bound          等式图上找 独立端 → 依赖端 的最短链，链上每条都 active
coefficient    沿链复合（斜率相乘、偏移累加），|ĉ−c|/|c| ≤ ε 且高次项 ≤ 1e−9
expected_dof   自由度总数 −（成员数 − active 图上的连通分量数）= 声明值
residual       33 个采样：中间关节按<b>模型</b>摆、依赖端按<b>契约</b>摆，读 efc_pos</div>

  <h3>KF3 · 可行运动一致性 —— 四层采样 + 3 条</h3>
  <p>把资产<b>瞬移</b>到上千个离散构型，每个 <code>mj_forward</code> 一次，
  用 <code>mj_geomDistance</code> 量部件对之间的距离。不积分轨迹、不算接触力。</p>
  <div class="f">采样   ① 逐自由度 33 点  ② 成对 9×9（只对包围盒可能相交的）
       ③ 声明状态全组合  ④ Halton 填充 256×d
       柜子 1,119 个构型（满网格 33³ = 35,937）；Halton 第 k 点是 k 的固定函数，<b>无随机种子</b>

forbidden_pair       该对通过 ⟺ 所有非豁免构型的最大穿透 ≤ 0.001 m；<b>按对计分不按样本</b>
required_contact     通过 ⟺ 该状态的构型里最近距离 ≤ 0.005 m
state_reachability   通过 ⟺ 该状态的构型里所有禁止对穿透 ≤ 0.001 m</div>

  <h3>打分</h3>
  <div class="f">score = 通过条数 / <b>不是 N/A 的条数</b>
分母 = 0 → 记 None（不是 1.0 也不是 0.0）

N/A = 工具评估不了这条声明（部件绑不上、关节无限位、slide 没有锚点…）
      <b>不计入分母，也不算通过</b></div>

  <h3>怎么审</h3>
  <ul>
    <li>先读 prompt，明确这个物体<b>本来该怎么动</b>。</li>
    <li>看四张环绕图（0°/90°/180°/270°，参考姿态，统一中性灰）。</li>
    <li>再看下面的判负清单，写下你自己的判断。<b>你写的不影响机器的分数</b>，单独导出。</li>
  </ul>
  <div class="n"><b>有些判负图上看不出来。</b>几毫米的轴套互穿、「从没碰到该锁的东西」这类缺失，
  以及只在中间构型出现的干涉（两个端点姿态都是干净的），四张静态图都显示不了——<b>这时以数值为准</b>。
  判负也可能是<b>契约写错</b>而不是资产做错，看着不对就直接写下来。</div>
</section>

<h2>逐个资产</h2>
{"".join(blocks)}
</main>
<footer>P3 运动学复核 · 10 个资产 · {n_fail} 条判负 · 40 张渲染图 ·
判分链路不调用任何模型 · 数据源 out/pilot/&lt;rid&gt;.json</footer>
<script>const TOTAL={total};{JS}</script>
</body></html>"""

out = pathlib.Path("docs/p3-review.html")
out.write_text(page, encoding="utf-8")
print(f"写出 {out} · {out.stat().st_size//1024} KB · {total} 个资产 · {n_fail} 条判负 · {total*4} 张图")
