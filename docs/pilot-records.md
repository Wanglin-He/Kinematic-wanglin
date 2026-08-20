# 试点替换名单：7 条

从 425 条（有已编译几何 + 有人工标注）中挑选。**只按类别和人工标注挑，没有看任何
资产的 link 结构** —— 按能不能过 Gate 来挑，就是拿测试集去迁就工具。

每条对应一个不同的 KF 子项，七条合起来把 KF1 的五个子项、KF2、KF3 全覆盖一遍。


## KF1·父子  —— Camcorder with flip-out screen

```
rec_camcorder_with_flipout_screen_b5753ec789b546b0a058ca52bbcd590d
```
- 人工判不满足：**KF1·父子**
- 几何：✓ 已物化   提示词：✓
- 提示词：A family camcorder with a soft-edged body shell, a cylindrical objective, and a left-side screen door with a shallow finger notch. The screen door rot

## KF1·类型  —— Air fryer with pullout basket

```
rec_air_fryer_with_pullout_basket_14ac4c8c9e35416b864382571fa27e62
```
- 人工判不满足：**KF1·类型**
- 几何：✓ 已物化   提示词：✓
- 提示词：A cylindrical matte-finish air fryer with a rounded body, a short basket drawer, and a top control tower carrying a temperature dial, a time dial, and

## KF1·轴向  —— Drill-press tilt table

```
rec_drillpress_tilt_table_0001
```
- 人工判不满足：**KF1·轴向, ★C5（无对应谓词）**
- 几何：✓ 已物化   提示词：✓
- 提示词：A realistic, highly detailed drill-press tilt table with a robust mounting structure, flat work surface, and convincing adjustment hardware. The table

## KF1·范围  —— Laptop clamshell

```
rec_laptop_clamshell_d201e66211d4426096f1625b7b4fc1cf
```
- 人工判不满足：**KF1·范围**
- 几何：✓ 已物化   提示词：✓
- 提示词：A rugged laptop with a thick lower chassis and a reinforced display housing. The base carries a keyboard and a large clickpad, and the screen panel is

## KF1·关节存在  —— Centrifugal juicer with articulated components

```
rec_centrifugal_juicer_with_articulated_components_8cf981d5e9bd4079bfcaa4a2e5ace079
```
- 人工判不满足：**KF1·关节存在, ★C5（无对应谓词）**
- 几何：✓ 已物化   提示词：✓
- 提示词：A stainless steel juicer with a deep base, a clear upper chamber, and a tall feed pusher centered over the basket housing. Two short side clamps rotat

## KF2·耦合  —— Glove compartment door

```
rec_glove_compartment_door_927c892914044efcb1cb608ac19db1b7
```
- 人工判不满足：**KF2·耦合**
- 几何：✓ 已物化   提示词：✓   **含 `<mimic>` 耦合**
- 提示词：A utility truck glove compartment with a top-hinged panel and twin limiter links. The fixed storage bin is boxy and shallow, and the front door is hin

## KF3·穿模  —— Ceiling fan

```
rec_ceiling_fan_a743cb8b7b834f689d8e18995b53306a
```
- 人工判不满足：**KF3·穿模**
- 几何：✓ 已物化   提示词：✓
- 提示词：A four-blade contemporary ceiling fan with a low-profile disc motor housing flush to the ceiling canopy, four angular aluminium blades on a central re

---

## 加上已经跑过的 3 条，一共 10 条

| | 类别 | 人工判不满足 |
|---|---|---|
| 已跑 | Stationary exercise bike | ★C5（无对应谓词） |
| 已跑 | Bicycle crankset and pedal assembly | 全部满足 |
| 已跑 | Stove top | ★C5（无对应谓词） |
| 新增 | Camcorder with flip-out screen | KF1·父子 |
| 新增 | Air fryer with pullout basket | KF1·类型 |
| 新增 | Drill-press tilt table | KF1·轴向, ★C5（无对应谓词） |
| 新增 | Laptop clamshell | KF1·范围 |
| 新增 | Centrifugal juicer with articulated components | KF1·关节存在, ★C5（无对应谓词） |
| 新增 | Glove compartment door | KF2·耦合 |
| 新增 | Ceiling fan | KF3·穿模 |
