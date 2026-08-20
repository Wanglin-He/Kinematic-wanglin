import json, pathlib, html
D = json.loads(pathlib.Path("/tmp/shots.json").read_text())
E = html.escape

CSS = """
:root{--ink:#1a2434;--muted:#66748a;--navy:#16324f;--teal:#0a7d76;--teal-s:#e4f4f0;
--paper:#f6f4ee;--card:#fffefb;--line:#dbd8d0;--gold:#a06c17;--gold-s:#faf1db;
--red:#a03f2e;--red-s:#fbe9e4;--green:#22715a;--green-s:#e7f4ee;
--mono:"SFMono-Regular",Consolas,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);line-height:1.6;background:var(--paper);
font-family:Inter,ui-sans-serif,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
code{font-family:var(--mono);font-size:.88em;padding:.06em .28em;border-radius:4px;
color:#075f5b;background:var(--teal-s)}
header{color:#fff;background:linear-gradient(118deg,#13304e,#0e6c66)}
.hero{max-width:1120px;margin:0 auto;padding:30px 24px 26px}
h1{margin:6px 0 0;font-family:Georgia,"Songti SC",serif;font-size:clamp(1.5rem,3vw,2.2rem);font-weight:600}
.eyebrow{color:#c9ece7;font-size:.72rem;font-weight:850;letter-spacing:.14em}
.sub{margin:8px 0 0;color:#d6e6ec;font-size:.93rem;max-width:760px}
main{max-width:1120px;margin:0 auto;padding:0 24px 60px}
section{margin-top:24px}
h2{margin:0 0 8px;color:var(--navy);font-family:Georgia,"Songti SC",serif;font-size:1.3rem}
p{margin:7px 0;font-size:.92rem}
.how{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:16px 18px;margin-top:18px}
.how h3{margin:14px 0 4px;font-size:.97rem;color:var(--navy)}
.how h3:first-of-type{margin-top:0}
.how ul{margin:6px 0;padding-left:1.15rem;font-size:.9rem}
.f{margin:7px 0;padding:9px 12px;border-radius:8px;background:var(--teal-s);color:#0b4744;
font-family:var(--mono);font-size:.78rem;line-height:1.6;white-space:pre-wrap;overflow-x:auto}
.n{margin:8px 0;padding:8px 12px;border-left:3px solid var(--gold);border-radius:6px;
background:var(--gold-s);color:#64450e;font-size:.85rem}
.n.b{border-left-color:var(--red);background:var(--red-s);color:#772f1f}
.bar{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:10px;flex-wrap:wrap;
padding:9px 24px;border-bottom:1px solid var(--line);background:#fffefbee;backdrop-filter:blur(8px)}
.bar b{font-size:.86rem;color:var(--navy)}
.bar .sp{flex:1}
button{padding:6px 13px;border:1px solid var(--navy);border-radius:7px;background:var(--navy);
color:#fff;font:inherit;font-size:.82rem;cursor:pointer}
button.g{background:#fff;color:var(--navy)}
button:hover{opacity:.86}
#tally{font-family:var(--mono);font-size:.8rem;color:var(--muted)}
.asset{margin-top:22px;border:1px solid var(--line);border-radius:14px;background:var(--card);overflow:hidden}
.asset>header{padding:12px 18px;border-bottom:1px solid var(--line);background:#f8f6f0}
.asset h3{margin:0;font-size:1.05rem;color:var(--navy)}
.rid{margin:2px 0 0;font-family:var(--mono);font-size:.68rem;color:var(--muted);word-break:break-all}
.prompt{margin:8px 0 0;padding:8px 11px;border-radius:7px;background:#f2efe7;color:#40506a;font-size:.83rem}
.scores{display:flex;gap:8px;flex-wrap:wrap;margin-top:9px}
.sc{padding:3px 11px;border:1px solid var(--line);border-radius:999px;background:#fff;
font-family:var(--mono);font-size:.8rem}
.sc.lo{color:var(--red);border-color:#dcaaa3;background:var(--red-s);font-weight:700}
.sc.hi{color:var(--green);border-color:#a8cebe;background:var(--green-s)}
.claim{padding:14px 18px;border-top:1px solid var(--line)}
.claim:first-of-type{border-top:none}
.ch{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
.ch .p{font-family:var(--mono);font-size:.88rem;font-weight:700;color:var(--red)}
.ch .s{font-family:var(--mono);font-size:.82rem;color:var(--muted)}
.mrow{margin:7px 0;padding:8px 12px;border-radius:7px;background:#f4f2ec;
font-family:var(--mono);font-size:.79rem;line-height:1.7;white-space:pre-wrap}
.mrow b{color:var(--red)}
.pics{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:9px 0}
.pic{border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#000}
.pic img{display:block;width:100%;height:auto}
.pic span{display:block;padding:5px 9px;background:#f4f2ec;color:var(--muted);font-size:.74rem;
font-family:var(--mono)}
.legend{font-size:.78rem;color:var(--muted);margin:2px 0 0}
.rev{margin-top:10px;padding:10px 13px;border:1px dashed #b9c4d2;border-radius:9px;background:#f7fafc}
.rev>b{font-size:.84rem;color:var(--navy)}
.opts{display:flex;gap:14px;flex-wrap:wrap;margin:7px 0}
.opts label{display:flex;align-items:center;gap:5px;font-size:.85rem;cursor:pointer}
textarea{width:100%;min-height:52px;padding:7px 9px;border:1px solid var(--line);border-radius:7px;
font:inherit;font-size:.86rem;resize:vertical;background:#fff}
input[type=text]{padding:5px 8px;border:1px solid var(--line);border-radius:6px;font:inherit;font-size:.84rem}
.saved{margin-left:8px;font-size:.75rem;color:var(--green)}
footer{margin-top:30px;padding:15px 24px;color:#cfe1e8;background:var(--navy);text-align:center;font-size:.8rem}
@media(max-width:640px){.pics{grid-template-columns:1fr}}
"""

JS = """
const KEY='p0p3-review-v1';
function load(){try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}}
function save(o){localStorage.setItem(KEY,JSON.stringify(o))}
function who(){return document.getElementById('who').value.trim()}
function collect(){
  const o={reviewer:who(),at:new Date().toISOString(),items:[]};
  document.querySelectorAll('[data-cid]').forEach(el=>{
    const cid=el.dataset.cid;
    const v=el.querySelector('input[type=radio]:checked');
    const t=el.querySelector('textarea').value.trim();
    if(v||t)o.items.push({id:cid,rid:el.dataset.rid,predicate:el.dataset.pred,
                          subject:el.dataset.subj,verdict:v?v.value:'',note:t});
  });
  return o;
}
function tally(){
  const o=collect();const n=o.items.length;
  const a=o.items.filter(x=>x.verdict==='agree').length;
  const d=o.items.filter(x=>x.verdict==='disagree').length;
  const q=o.items.filter(x=>x.verdict==='unsure').length;
  document.getElementById('tally').textContent=
    `已审 ${n}/${TOTAL}  ·  同意 ${a}  不同意 ${d}  存疑 ${q}`;
}
function persist(){
  const st=load();
  document.querySelectorAll('[data-cid]').forEach(el=>{
    const cid=el.dataset.cid;
    const v=el.querySelector('input[type=radio]:checked');
    st[cid]={v:v?v.value:'',t:el.querySelector('textarea').value};
  });
  st.__who=who();save(st);tally();
}
function restore(){
  const st=load();
  if(st.__who)document.getElementById('who').value=st.__who;
  document.querySelectorAll('[data-cid]').forEach(el=>{
    const s=st[el.dataset.cid];if(!s)return;
    if(s.v){const r=el.querySelector(`input[value="${s.v}"]`);if(r)r.checked=true}
    if(s.t)el.querySelector('textarea').value=s.t;
  });
  tally();
}
function download(){
  const blob=new Blob([JSON.stringify(collect(),null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download=`p0p3-review-${who()||'anon'}.json`;a.click();
}
function copyJSON(){
  navigator.clipboard.writeText(JSON.stringify(collect(),null,2))
    .then(()=>{const b=document.getElementById('cp');const o=b.textContent;
      b.textContent='已复制 ✓';setTimeout(()=>b.textContent=o,1400)});
}
document.addEventListener('DOMContentLoaded',()=>{
  restore();
  document.addEventListener('input',e=>{if(e.target.closest('.rev')||e.target.id==='who')persist()});
  document.addEventListener('change',e=>{if(e.target.closest('.rev'))persist()});
});
"""

def mrow(sh):
    m, t = sh["measured"], sh["threshold"]
    lines = []
    if m: lines.append("实测   " + " · ".join(f"{k} = {v}" for k, v in m.items()))
    if t: lines.append("阈值   " + " · ".join(f"{k} = {v}" for k, v in t.items()))
    ev = {k: v for k, v in (sh["evidence"] or {}).items()
          if k in ("worst_configuration", "first_failing_configuration",
                   "closest_configuration", "anchor_world", "axis_world",
                   "nearest_declared_ancestor", "body_chain_upward", "part", "relation")}
    if ev: lines.append("证据   " + " · ".join(f"{k} = {v}" for k, v in ev.items()))
    return "\n".join(lines)

total = sum(len(a["shots"]) for a in D)
parts_html = []
for a in D:
    p = a["profile"]
    def sc(k):
        v = p.get(k)
        if v is None: return f'<span class="sc">{k} —</span>'
        cls = "hi" if v >= 0.999 else ("lo" if v < 0.75 else "")
        return f'<span class="sc {cls}">{k} {v:.2f}</span>'
    blocks = []
    for i, sh in enumerate(a["shots"]):
        cid = f'{a["rid"]}::{sh["predicate"]}::{sh["subject"]}'
        legend = "红 = " + sh["parts"][0] if sh["parts"] else ""
        if len(sh["parts"]) > 1: legend += " · 蓝 = " + sh["parts"][1]
        legend += " · 其余半透明是为了看见内部"
        blocks.append(f"""
      <div class="claim">
        <div class="ch"><span class="p">{E(sh["predicate"])}</span><span class="s">{E(sh["subject"])}</span></div>
        <div class="mrow">{E(mrow(sh))}</div>
        <p class="legend">{E(legend)}</p>
        <div class="pics">
          <div class="pic"><img src="data:image/png;base64,{sh["ref"]}" alt=""><span>参考姿态</span></div>
          <div class="pic"><img src="data:image/png;base64,{sh["fail"]}" alt=""><span>{E(sh["caption"] or "判负构型")}</span></div>
        </div>
        <div class="rev" data-cid="{E(cid)}" data-rid="{E(a["rid"])}"
             data-pred="{E(sh["predicate"])}" data-subj="{E(sh["subject"])}">
          <b>人工审核</b>
          <div class="opts">
            <label><input type="radio" name="v{len(parts_html)}_{i}" value="agree"> 同意机器</label>
            <label><input type="radio" name="v{len(parts_html)}_{i}" value="disagree"> 不同意（机器误报）</label>
            <label><input type="radio" name="v{len(parts_html)}_{i}" value="unsure"> 存疑 / 看不出来</label>
          </div>
          <textarea placeholder="写下你的理由，或图上看不出来的地方…"></textarea>
        </div>
      </div>""")
    if not blocks:
        blocks.append("""
      <div class="claim"><p style="color:var(--green);margin:0"><b>零条判负</b> —— 这个资产没有可审核的判负条目。</p>
      </div>""")
    parts_html.append(f"""
    <div class="asset">
      <header>
        <h3>{E(a["zh"])}</h3>
        <p class="rid">{E(a["rid"])}</p>
        {'<p class="prompt">Prompt：' + E(a["prompt"][:600]) + '</p>' if a["prompt"] else ''}
        <div class="scores">{sc("KF1")}{sc("KF2")}{sc("KF3")}
          <span class="sc">{a["n_claims"]} 条声明</span><span class="sc">{a["n_na"]} 条 N/A</span>
          <span class="sc">{len(a["shots"])} 条判负</span></div>
      </header>
      {"".join(blocks)}
    </div>""")

page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>P3 运动学判负复核</title>
<style>{CSS}</style></head><body>
<header><div class="hero">
  <div class="eyebrow">EVO-ARTICRAFT · P3 · 人工复核</div>
  <h1>运动学判负复核 · {total} 条 · 10 个资产</h1>
  <p class="sub">机器已经判完，这一页是给人复核用的：每一条判负都带上它是<b>用哪个数、在哪个构型上</b>
  决定的，以及那个构型的渲染图。你的判断独立于机器，填在每条下面。</p>
</div></header>

<div class="bar">
  <b>审核人</b><input type="text" id="who" placeholder="你的名字" size="10">
  <span id="tally"></span><span class="sp"></span>
  <button class="g" id="cp" onclick="copyJSON()">复制 JSON</button>
  <button onclick="download()">导出 JSON</button>
</div>

<main>
<section class="how">
  <h2>先说清楚：这些判负是怎么定出来的</h2>
  <h3>判分不看图</h3>
  <p>页面上的每个分数都由 <code>mj_geomDistance</code> 和几条算术式决定，<b>渲染不在判分链路上</b>。
  图是评完分之后按结果里记录的构型补渲的，<b>它改不了任何分数</b>，只是让人能看见同一个东西。</p>
  <div class="f">score = 通过的条数 / 不是 N/A 的条数
分母 = 0 → 记 None（<b>不是 1.0 也不是 0.0</b>）

KF1  关节配置保真度   9 条：父件 · 类型 · 自由度构成 · 轴向（语义/可行）·
                          锚点 · 行程 · 刚性随动 · 尺度
KF2  机械耦合保真度   4 条：约束存在 · 系数 · 剩余自由度 · 残差
KF3  可行运动一致性   3 条：禁止对 · 必须接触 · 状态可达
                     在 1,119 个确定性采样构型上评估（无随机种子）</div>

  <h3>三个数字的含义</h3>
  <ul>
    <li><b>实测 / 阈值</b>——判负就是实测越过了阈值，两个数都摆出来，可以直接质疑阈值。</li>
    <li><b>构型</b>——像 <code>display_hinge=+1.575</code> 或 <code>halton[122]</code>，
        是采样计划里的确定性编号，<b>复跑会得到同一个</b>。</li>
    <li><b>N/A</b>——工具评估不了的声明记 N/A，不计入分母。<b>不是通过。</b></li>
  </ul>

  <h3>怎么审</h3>
  <ul>
    <li>先读 prompt，明确这个物体<b>本来该怎么动</b>。</li>
    <li>看两张图：左边是参考姿态，右边是判负的那个构型，<b>同一个相机</b>，只差关节角。</li>
    <li>红 / 蓝是这条判负点名的部件，<b>其余部件被画成半透明</b>——因为多数穿透发生在内部，
        实心外观图看起来一切正常。</li>
    <li>选「同意 / 不同意 / 存疑」，写下理由。<b>你的判断不会影响机器的分数</b>，
        它单独导出，用来算这套检查的误报率。</li>
  </ul>
  <div class="n b"><b>有几类问题图上本来就看不出来。</b>轴套之间几毫米的互穿、
  锁柄「从没碰到该锁的东西」这种缺失类判负，肉眼没法从渲染图确认——<b>这时以数值为准，
  选「存疑」并写明</b>，不要因为图上看着正常就判机器误报。</div>

  <h3>这批数字的两个前提</h3>
  <ul>
    <li><b>这是诊断结果，不是最终得分。</b>严格绑定下这十条<b>全部被 Gate 拒绝</b>（部件粒度与提示词不一致），
        表中数字是打开诊断模式、对能绑上的部件评估、其余记 N/A 得到的。</li>
    <li><b>契约还没有第二个人核过。</b>判负也可能是契约写错——
        例如 <code>range_and_reference</code> 触发 7/10，契约里四处写 <code>1.5708</code>（π/2），
        资产分别建成 2.15 / 1.05 / 1.05 / 1.20，<b>一个都没对上</b>，
        很可能是契约作者在提示词没给数字时的默认填法。<b>这类请判「不同意」并注明「契约问题」。</b></li>
  </ul>
</section>

<section>
<h2>逐条复核</h2>
{"".join(parts_html)}
</section>
</main>
<footer>P3 运动学判负复核 · {total} 条判负 · 84 张渲染图 · 判分链路不调用任何模型 ·
数据源 out/pilot/&lt;rid&gt;.json</footer>
<script>const TOTAL={total};{JS}</script>
</body></html>"""

out = pathlib.Path("docs/p3-review.html")
out.write_text(page, encoding="utf-8")
print(f"写出 {out} · {out.stat().st_size//1024} KB · {total} 条判负 · {total*2} 张图")
