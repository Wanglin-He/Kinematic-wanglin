# Kinematic — Evo-Articraft P0 与 P3

**P0**（提示词契约）与 **P3**（运动学保真度 KF1/KF2/KF3）的实现。

设计与实现都在资产之前冻结：代码先建好、冻住，再拿真实的 Articraft 模型当考卷。反过来做——
按已知答案去调评测器——跑得再好也证明不了什么。

```bash
uv run --no-project --with pyyaml --with pytest python -m pytest -q
uv run --no-project --with pyyaml python -m evo_p0p3.p0.cli validate contracts/
```

## 状态

| | 状态 |
|---|---|
| P0 schema / loader / 准入检查 A1–A16 | ✅ 33 个测试 |
| P3 KF1 关节配置保真度 | 🚧 谓词推导中 |
| P3 KF2 机械耦合保真度 | 🚧 |
| P3 KF3 可行运动一致性 | 🚧 |
| 金标准资产（1 正确 + 11 缺陷） | 🚧 |
| 10 个真实模型的验证 | ⏳ 等 P3 冻结 |

## 模块

| 文件 | 作用 |
|---|---|
| `evo_p0p3/p0/schema.py` | 冻结契约的类型定义 |
| `evo_p0p3/p0/loader.py` | YAML → 类型化契约。**形状错误在这里抛**，带字段路径 |
| `evo_p0p3/p0/admission.py` | A1–A16 准入检查。**语义问题在这里返回报告**，不抛异常 |
| `evo_p0p3/p0/cli.py` | `p0 validate` |
| `contracts/gold_cabinet.yaml` | 金标准契约：slide + hinge + 刚性随动 |

两层分开是有意的：一个格式错误的文件**读不了**，一个格式正确但引用了未声明名字的契约**读得了但检查不了**——
失败时机不同，输出形状也该不同。

## 几条设计决定，以及为什么

**每个名字都必须能解析到几何。** 上一轮把散文部件名去匹配代码标识符，结果 70 次报警里 63 次误报、
其中 42 次来自「找不到零件」——但 7 个真阳性里也有 4 个来自同一个标志。信号和噪声是同一个事件，
配对 bootstrap 证明写更好的匹配器救不了。所以规则 A1：任何字段引用的 ID 必须在 `required_parts`
或 `joints` 里声明过，**在生成之前**就把「契约写错」这个解释消掉，让这个信号只剩一种含义。

**实例枚举，不用 `count`。** 从来没有任何地方规定 `{id: drawer, count: 3}` 怎么展开成
`drawer_1..3`，也没规定下标顺序。`group` 标签保留集合寻址的便利，但成员是列出来的，展开可校验。

**能用关系表达的就不写坐标。** URDF 没有语义方向，只有数字；两个建模时差 90° 的相同柜子是同一个物体。
要求生成器按规范朝向输出，会把「约定违规」变成「运动学失败」——比较多个生成器时，测到的是谁读了坐标系
说明。所以轴向有三种写法：`semantic`（对重力解析，免费）、`relational`（从两个部件的几何解析，跟着
资产走）、`numeric`（仅当 `canonical_frame.front` 已锚定，否则**报 N/A 不报失败**）。

**slide 的 anchor 恒为 N/A。** 平移没有中心：MuJoCo 里 slide 关节的 `pos` 设成任何值，子体位姿
逐比特相同。写了就是不可检的声明，准入直接拒绝。hinge 有两条**互为反面**的谓词——门要求几何全在
轴线平面一侧，齿轮要求几何绕轴对称——用门的谓词去判齿轮会必然判负。

**`allowed` 拆成 `required` / `permitted`。** 旧写法从未说明它断言什么：是「这里接触不算违规」
（那它永远不会失败），还是「必须接触」（那它需要容差和状态谓词）。拆开逼作者在写的时候就决定。

**所有阈值进契约。** `tolerances.digest()` 随每份结果输出——指不出来的冻结不算冻结。

## 测试的写法

每个测试只破坏金标准契约的**一处**，断言对应规则被触发。**触发不了的规则等于没有**：
上一轮做过一个故障注入，把关节原点整体挪开一整条连杆对角线，检测率是零——因为关节原点就是子体坐标系
原点，挪它会把几何一起带走。被检的量如果是从被检字段推出来的，它就不可能失败。

## 数据

真实资产在 `articraft-data/cache/record_materialization/`：546 条已物化的 `model.urdf` + OBJ 网格，
其中 **425 条同时有人工标注**。加载方式（实测 545/546 成功）：

```
① 往 <robot> 注入 <mujoco><compiler discardvisual="false" inertiafromgeom="true"
   balanceinertia="true" strippath="false" fusestatic="false"/></mujoco>
② 给缺 <inertial> 的 link 补 dummy —— 这是合成值，标记 inertia_synthesized，绝不进任何分数
```

零个 `<collision>`，所有 geom `contype=0`，所以 `mjData.contact` 恒为空。距离一律走
`mj_geomDistance`——它不看 contype/conaffinity，也不受父子过滤影响。
