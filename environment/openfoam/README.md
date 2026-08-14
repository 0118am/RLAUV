# AUV 6-DOF 水动力系数 OpenFOAM 部署

本目录用于从强制振荡 CFD 载荷中辨识完整的 `6×6` 附加质量矩阵
`M_A`、线性阻尼矩阵 `D_L` 和二次阻尼矩阵 `D_Q`。目标版本锁定为
**OpenCFD OpenFOAM v2512**；Foundation 版与其他 OpenCFD 版本的求解器和动态网格
字典不能直接混用。

## 当前几何门禁

当前生产输入来自 [`验证机装配体.STEP`](验证机装配体.STEP)，源文件 SHA-256 为
`9777be1c028d8ebb18f61118466d17671aee1f5860ea8144717c50bc65d6ba07`。处理流程不会
修改源 STEP，并由
[`geometry/verification_assembly_repair.json`](geometry/verification_assembly_repair.json)
记录源指纹并锁定经过复核的壳编号：保留外壳、主压力筒、螺丝孔、舱盖/窗口、推进器安装支架、
尾翼以及 8 组外置螺旋桨/桨毂；删除压力舱内框架/拉杆及电机线缆、紧固件细节，并用 8 个
单实体轴对称光滑包络替换详细电机。包络包含半径 `16.5 mm` 主壳、经 STEP 复核的多段鼻罩
和锁桨轴；主壳向支架侧延伸 `5.5 mm`。它不是逐细节 CAD 复刻，而是保留湿表面主要截面积
的水动力简化。

生产工况是“全装实机、推进器不施加推力、转子相对艇体静态锁定”。STEP 实测电机轴半径
为 `2.0 mm`；包络内的细轴改为半径 `2.05 mm`，从桨毂前端多伸 `0.5 mm` 并连续进入鼻罩。
额外 `0.05 mm` 只位于名义配合孔内。19 点旋转剖面对原 B-spline 最大径向误差
`0.0752 mm`；生成包络体积 `36861.963 mm³`，相对 8 台原详细电机均约低 `0.2811%`，
通过 `1%` 门禁。每组包络还必须与 mount/hub/prop 分别产生至少 `1 mm³` 的实体交叠；
实测最小值约为 `376.2/41.47/234.92 mm³`。
8 台水平/竖直电机的实际倾轴和端点均由 STEP 逐台测量并写入 selection report，不硬编码
世界轴。若实际试验是自由转桨，必须另建转子动力学工况，不能把本静态锁桨矩阵直接当作自由转结果。

STEP 坐标先按
`(x_body,y_body,z_body)=(z_step,x_step,y_step)` 映射，再加
`(+1.306,-0.061,-2.385) mm`；这是质量报告中坐标系1质心
`(-1.306,+0.061,+2.385) mm` 的相反数，使输出原点成为实测 COM。该平移是固定的质量
属性修正，不是按几何包围盒自动居中，后续转动中心仍为 `(0,0,0)`。

STEP 顶层装配表示的放置点为 `(0,0,0)` 且方向为单位轴，但 AP203 文件没有保留名为
“坐标系1”的实体或它与顶层放置的关系。因此“质量报告坐标系1原点等于 STEP 全局原点”
不能仅从 STEP 严格证明；这里依据同一 SolidWorks 装配体、轴向/惯性次序及已标号推进器
位置作高置信采用，并在 selection report 中显式记录该假设。

shell 257 是体坐标上侧的大型闭实体，STEP 实测闭实体体积为 `2.674187522 L`。结合其
几何、位置以及实机确有浮力材料这一事实，将它标注为 **waterproof closed-cell main
buoyancy material**；由于扁平 STEP 没有保留原 SolidWorks 零件名，身份状态明确写为
`high-confidence geometry/placement inference`，不是声称从 STEP 名称严格证明。repair
会实测并校验它的体积和 STEP 包围盒，且 shell 257 始终显式保留。它已经通过几何并集
自然计入排水体积，绝不再把 `2.674187522 L` 数值加到 `11.304505834 L` 目标上，以免双计。

主压力筒 shell 30 的外径约 `130 mm`、长 `300 mm`。shell 42/43 实际是含服务孔的开放
环形端部法兰；它们虽与筒壁接触，实体公共体积为零，直接做 exterior wrap 会从孔洞灌入
内腔。正式修复从 shell 30 实测的 `Rin=60 mm`、`Rout=65 mm`、轴线和两个筒口位置
派生两片密封盘：每片厚 `1 mm` 并以筒口为中心沿轴各跨 `0.5 mm`，半径只向筒壁内增加
`0.5 mm` 到 `60.5 mm`。这样每片同时与筒壁和对应法兰形成正实体交叠，但仍远在外径以内，
不会把外侧法兰凹槽错误封成排水体积。它们代表真实端盖/O-ring 的防水边界，而不是把整个
压力舱填成实心圆柱；舱内容积由表面闭合自然排除，数值上不另加任何目标体积。

独立诊断先用 `R=60 mm` 相切薄盘确认了筒口位置；正式版使用更稳健的
`R=60.5 mm` 正交叠，并已从最终 repair 产物完成 `0.5 mm` wrap。正式结果为
`11.460906875 L`，相对目标 `+1.38353%`，通过原定
`11.304505834 L ±2%` 门禁；输出为单连通水密流形，boundary/non-manifold 边均为零。
OpenFOAM v2512 进一步确认它无非法三角形、法向一致且无自相交。随后绕 COM 原点统一
缩放 `0.001` 得到正式米制输入，范围为 `0.562 × 0.4025 × 0.191 m`。

当前正式输入只来自 `geometry/validated_locked_rotor_v1/`，面数、哈希和范围均从该目录的
重建报告读取。

正式选择明细见
[`selection_report.json`](geometry/validated_locked_rotor_v1/selection_report.json)，
包络拓扑与体积门禁见
[`wetted_body_mm.json`](geometry/validated_locked_rotor_v1/wetted_body_mm.json)，
米制缩放、边界和哈希见
[`wetted_body_m.provenance.json`](geometry/validated_locked_rotor_v1/wetted_body_m.provenance.json)。
这些均为可再生的大文件/报告，默认不进 Git。

严格拓扑通过不等于几何已经物理收敛。voxel 报告中的 Euler 特征/genus
可同时包含真实孔洞和 `0.5 mm` 分辨率下残留的细缝/通道；数值只能在新包络生成后
报告，不沿用作废表面的统计。仍须在 ParaView 中人工核对外形与孔洞连通性，并通过网格、
时间步和外域收敛研究后才能发布水动力矩阵。

## 数学定义

体坐标采用项目现有的 COM 原点、FLU 右手系：x 向前、y 向左、z 向上。

```text
nu    = [u, v, w, p, q, r]^T
tau_h = [X, Y, Z, K, M, N]^T
```

CFD 输出是流体作用于艇体的体坐标载荷，拟合模型为：

```text
tau_h = -M_A nu_dot - C_A(nu; M_A) nu
        -D_L nu - D_Q (abs(nu) .* nu) + error
```

矩阵使用“正阻力系数”符号；例如正 surge 应产生负 `X`。每次只激励一个自由度，
但始终采集全部六个力/矩响应，因此每个试验得到三张矩阵的一整列，六个自由度共同
覆盖全部 `36` 个耦合项。每个完整周期先重采样到相同的均匀相位网格，避免自适应时间步
把某些相位或算例意外加权；随后半周期奇对称投影消除静态偏置和对速度为偶函数的惯性
科氏项，再联合拟合加速度、速度和 `abs(nu).*nu`。周期中存在大时间缺口时直接拒绝，
不跨缺口插值。

这里的二次矩阵严格指
`tau_i = -sum_j D_Q[i,j] * abs(nu_j) * nu_j`。它不包含
`nu_j*abs(nu_k), j != k` 的全部混合乘积；后者是 `6×6×6` 的 216 参数张量，
不是用户要求的 `6×6` 矩阵，也不能由单轴试验辨识。

矩阵行列的 SI 单位如下（角度按 rad，视为无量纲）：

| 矩阵 | 力行/平移列 | 力行/转动列 | 矩行/平移列 | 矩行/转动列 |
|---|---:|---:|---:|---:|
| `M_A` | kg | kg·m | kg·m | kg·m² |
| `D_L` | kg/s | kg·m/s | kg·m/s | kg·m²/s |
| `D_Q` | kg/m | kg·m | kg | kg·m² |

## 默认试验设计

部署生成 `6 × 2 × 2 = 24` 个强制振荡算例：

- 平移位移幅值：`0.01`, `0.025` m；
- 转动角幅值：`2`, `5` deg（写入 OpenFOAM 前转换为 rad）；
- 频率：`0.75`, `1.5` Hz；
- 每个算例丢弃前 `3` 周期，拟合后续 `5` 周期；
- 水密度 `997 kg/m³`，运动黏度 `1.004e-6 m²/s`；
- 单相、无重力、无自由液面、推进器零推力且桨静态锁定，URANS `kOmegaSST`；
- `forces` 同时保留压力、黏性和总载荷，力矩参考点固定为初始 COM `(0 0 0)`。

小幅运动使用固定外边界加 `displacementLaplacian` 变形网格。若后续提高位移/角度
导致网格层畸变，应升级为 body-fitted overset 内域，而不是继续放宽网格质量限制。

## 目录

```text
environment/openfoam/
  config.json               # 流体与 24 个运动试验的配置
  case_template/            # v2512 完整基础算例
  geometry/                 # STEP 选择配置；processed/ 为可再生几何输出
  tools/                    # STEP 筛选、电机简化、voxel 包络、缩放和 STL 审计
  analysis/                 # forces 解析、坐标变换、矩阵拟合和诊断
  build_mesh.py             # 严格几何门禁、网格构建、质量检查和算例分发
  generate_cases.py         # 生成所有单轴强制振荡算例
  run_cases.py              # 有界 MPI/多算例执行器
  cases/                    # 生成物，不进 Git
  results/                  # JSON/CSV 矩阵和报告
```

## 环境

本仓库支持无需 `sudo` 的项目内离线部署。安装器会校验已下载 Debian 包的
SHA-256，将 OpenCFD v2512 的二进制、源码、教程和三个缺失的数值库解压到
`environment/openfoam/.runtime/openfoam2512/`；该目录约 `594 MiB` 且已被 Git 忽略：

```bash
environment/openfoam/install_local.sh --package-dir /path/to/downloaded/debs
source environment/openfoam/env.sh
python3 environment/openfoam/tools/check_environment.py --strict --min-api 2512
environment/openfoam/verify_local_install.sh
```

若不带参数运行启动器，会打开一个已加载环境的交互式 shell；也可直接执行一个
OpenFOAM 命令：

```bash
environment/openfoam/launch_openfoam.sh
environment/openfoam/launch_openfoam.sh --version
environment/openfoam/launch_openfoam.sh foamEtcFile -show-api
environment/openfoam/launch_openfoam.sh pimpleFoam -help
```

`env.sh` 的查找顺序是 `AUV_OPENFOAM_BASHRC`、项目内 `.runtime`、系统 `/usr/lib`
和 `/opt`。如需把二进制放在别处，可在安装和加载时设置同一个
`AUV_OPENFOAM_ROOT`。

### ParaView 图形界面

本部署使用与 OpenFOAM v2512 配置一致的 Kitware ParaView 6.0.1。无需 `sudo`，
下载官方 Linux 包后安装到同一个被忽略的 `.runtime`：

```bash
curl -fL -o /tmp/ParaView-6.0.1-MPI-Linux-Python3.12-x86_64.tar.gz \
  https://www.paraview.org/files/v6.0/ParaView-6.0.1-MPI-Linux-Python3.12-x86_64.tar.gz
cd /tmp && apt-get download libxcb-cursor0=0.1.1-4ubuntu1
cd -
environment/openfoam/install_paraview.sh
environment/openfoam/verify_paraview_install.sh
```

直接打开 ParaView，或通过内置 OpenFOAM reader 打开一个算例：

```bash
environment/openfoam/launch_paraview.sh
environment/openfoam/launch_paraview.sh --case /absolute/path/to/case
```

当前二进制 OpenFOAM 包没有版本绑定的 `ParaFoamReader` 插件，因此案例入口显式使用
`paraFoam -vtk` 和 `.foam` 文件。这是 ParaView 自带的 OpenFOAM reader，不需要编译
插件；启动包装器会隔离 OpenFOAM 与 ParaView 的 Qt/MPI 动态库，避免混用。

已有系统安装时也可直接加载并检查：

```bash
source environment/openfoam/env.sh
python3 environment/openfoam/tools/check_environment.py --strict --min-api 2512
```

也可使用官方镜像 `opencfd/openfoam-default:2512`：

```bash
environment/openfoam/run_in_docker.sh bash
```

系统安装方法以 OpenCFD 官方 Linux 安装页为准。

## 执行顺序

1. 安装固定版本的 OCCT Python 运行时，然后从 STEP 生成经筛选和电机简化的体坐标
   mm 制中间 STL。脚本会自动记录 STEP 指纹，但不把用户提供的 SHA 当作运行门禁；
   输出仍是相交多实体，只能作为下一步的输入。所有新产物与当前正在运行的
   `environment/openfoam/cases` 物理隔离：

   ```bash
   environment/openfoam/install_cad_tools.sh

   python3 environment/openfoam/tools/repair_step.py \
     environment/openfoam/验证机装配体.STEP \
     environment/openfoam/geometry/validated_locked_rotor_v1/selected_body_mm.stl \
     --config environment/openfoam/geometry/verification_assembly_repair.json \
     --report environment/openfoam/geometry/validated_locked_rotor_v1/selection_report.json \
     --force
   ```

   中间选择 STL 仍保留零件间相交面，因此不能直接进入 OpenFOAM。selection report 会记录
   每组锁桨连接轴的 STEP/body 轴线、桨中心和三项实体交叠；
   下一步 voxel exterior wrap 会重建表面，最终必须由 `surfaceCheck -checkSelfIntersection`
   证明中间缺陷已经清除。

2. 以 `0.5 mm` 体素构造单一外部湿表面。`4.1 mm` 连接轴横跨约 `8.2` 个体素，
   `83 mm` 桨盘横跨约 `166` 个体素，足以建立首个全装锁桨外形；解析曲面抽样得到叶片
   厚度第 5 百分位约 `0.385 mm`，所以该分辨率会圆钝最薄约 5% 的尖缘。必须用单组转子
   `0.5/0.25 mm` 局部裁剪对照和 snappy level 6↔7 对照验证，不能把全局体积通过当作
   叶片保真。该分辨率需要约
   `3.60e8` 个体素，峰值内存和
   临时磁盘需求明显高于普通 STL 处理；不要在资源不足时无依据地增大体素尺寸。这里必须
   使用项目内 ParaView 6.0.1 的 `pvpython`，并清除 OpenFOAM/Python 环境变量，避免系统
   Python 的 SciPy/NumPy ABI 不兼容：

   ```bash
   env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
     environment/openfoam/.runtime/paraview-6.0.1/bin/pvpython \
     environment/openfoam/tools/voxel_wrap.py \
     environment/openfoam/geometry/validated_locked_rotor_v1/selected_body_mm.stl \
     environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_mm.stl \
     --voxel-size 0.5 --closing 1 --max-voxels 450000000 \
     --expected-volume 11304505.834 --volume-relative-tolerance 0.02 \
     --json environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_mm.json \
     --force

   environment/openfoam/launch_openfoam.sh surfaceCheck \
     -checkSelfIntersection -outputThreshold 0 \
     environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_mm.stl
   ```

3. 仅在 mm 表面通过上述严格门禁后，以流式方式绕已修正的 COM 原点缩放到 m。这个工具不会为
   `16M+` 面表面构造内存巨大的 VTK 边图：

   ```bash
   python3 environment/openfoam/tools/scale_binary_stl.py \
     environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_mm.stl \
     environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_m.stl \
     --scale 0.001 \
     --provenance \
     environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_m.provenance.json \
     --force

   environment/openfoam/launch_openfoam.sh surfaceCheck \
     -checkSelfIntersection -outputThreshold 0 \
     environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_m.stl
   ```

4. 最终文件已经是 m 制 body-FLU 湿表面，因此用 `--prepared-input` 跳过再次缩放和
   大型 VTK 拓扑构图；缩放 provenance 已记录输入/输出 SHA，建网入口也会重新计算并在
   运行摘要中打印实际输入 SHA，不要求用户手工提供哈希门禁。
   OpenFOAM 严格表面门禁和排水体积门禁仍强制执行。先用 dry-run 核对命令，
   再在独立 `cases_locked_rotor_v1` 中建共享网格；不得指向正在使用的 `environment/openfoam/cases`：

   ```bash
   python3 environment/openfoam/build_mesh.py \
     environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_m.stl \
     --prepared-input \
     --repair-report environment/openfoam/geometry/validated_locked_rotor_v1/selection_report.json \
     --expected-displaced-volume-m3 0.011304505834 \
     --mesh-volume-relative-tolerance 0.055 \
     --cases-dir environment/openfoam/cases_locked_rotor_v1 \
     --provenance \
     environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_m.provenance.json \
     --mesh-only --dry-run

   source environment/openfoam/env.sh
   python3 environment/openfoam/build_mesh.py \
     environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_m.stl \
     --prepared-input \
     --repair-report environment/openfoam/geometry/validated_locked_rotor_v1/selection_report.json \
     --expected-displaced-volume-m3 0.011304505834 \
     --mesh-volume-relative-tolerance 0.055 \
     --cases-dir environment/openfoam/cases_locked_rotor_v1 \
     --provenance \
     environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_m.provenance.json
   ```

   入口依次自动记录 SHA，再执行 `surfaceCheck -checkSelfIntersection`、`blockMesh`、
   `surfaceFeatureExtract`、`snappyHexMesh -overwrite` 和
   `checkMesh -allGeometry -allTopology -meshQuality`。网格硬门槛要求：两个工具日志均以
   `End` 正常结束，snappy/checkMesh 最后一组 `meshQualityDict` 配置阈值计数全部为零，
   核心拓扑项目均为 `OK`，只有一个流体连通域，最小单元体积和流体总体积均为正。
   `-allGeometry/-allTopology` 额外报告的重复面、非连续共享点、较保守的凹度、行列式或
   插值权重诊断会完整写入 `mesh_quality_audit.json` 作为警告；它们不会在仍满足实际
   `meshQualityDict` 阈值时，仅因汇总行出现 `Failed N mesh checks` 而被误判为硬失败。
   `FOAM FATAL`、日志截断、负/零体积、缺失核心证据或任一配置阈值超限仍立即失败。随后用
   `blockMesh` 外域体积减去 `checkMesh` 流体总体积，要求 snappy 排除体积相对
   `0.011304505834 m³` 的误差不超过 `5.5%`，再生成 24 个
   运动算例及一个静止 baseline，并以相对链接分发共享 `polyMesh`。表面门禁日志写入
   `environment/openfoam/geometry/validated_locked_rotor_v1/surfaceCheck.log`，网格日志位于
   `environment/openfoam/cases_locked_rotor_v1/mesh_case/logs/`。`--allow-dirty` 只供调试，发布流程禁止使用。

   网格生成器从 selection report 的 8 条实际倾轴自动写入局部加密区：
   桨盘加密圆柱半径 `45 mm`、轴向 `30 mm`，连接轴加密半径 `6 mm`，以及覆盖鼻罩/主壳
   的半径 `20 mm` 电机圆柱均到 level 7；电机圆柱从主壳起点向桨侧多覆盖 `15 mm`，与
   桨盘区重叠而不留粗网带。
   （本外域基础网格下约 `0.651 mm`）；每个桨中心周围半径 `50 mm` 的各向
   `searchableSphere` 近场加密到 level 5（约 `2.60 mm`）。锁桨随艇运动没有固定的
   电机轴向尾流，因此这里不生成轴向 wake cylinder；桨盘和连接轴也不使用硬编码世界轴。
   当前全局表面 level 4 约 `5.21 mm`，单独使用它无法解析 `4.1 mm` 连接轴。8 个桨盘和
   8 个电机区估算约 `7.6M` 个 level-7 单元（未扣实体/重叠且未含过渡层），因此
   `maxLocalCells` 与 `maxGlobalCells` 均设为 `16000000`，避免 serial snappy 被旧的
   150 万 local 上限截断；最终仍以实际 snappy 日志和 checkMesh 为准。

5. 先跑一个平移和一个转动冒烟算例，再执行全批次：

   ```bash
   python3 environment/openfoam/run_cases.py --cases-dir environment/openfoam/cases_locked_rotor_v1 --np 1 --only 'u_amp0p010m_f0p75hz'
   python3 environment/openfoam/run_cases.py --cases-dir environment/openfoam/cases_locked_rotor_v1 --np 1 --only 'p_amp2p0deg_f0p75hz'
   python3 environment/openfoam/run_cases.py --cases-dir environment/openfoam/cases_locked_rotor_v1 --np 4 --jobs 1 --resume
   ```

   `--np × --jobs` 不应超过可用核心数；并行求解前应先用一个案例确认 MPI、动态网格
   和磁盘空间。执行器不会把只用于建网格的 `mesh_case` 当作求解案例。模板保留最后
   `4` 个场写出相位，但逐时间步的 force/moment 历史不会被清除；全矩阵拟合不需要
   reconstruct 全部处理器场，只对选定的收敛/可视化工况按需重建。
   首个时间步使用每周期名义最大时间步的 `1/20`，随后由 `maxCo` 自适应增长到
   `maxDeltaT`；这是为了避免初始零流场下旋转网格的首步 Courant 数尖峰。

6. 收集全部 OpenCFD v2512 `postProcessing/forces/*/{force,moment}.dat` 数据并拟合
   三张完整矩阵：

   ```bash
   python3 -m environment.openfoam.analysis \
     --cases-root environment/openfoam/cases_locked_rotor_v1 \
     --config environment/openfoam/config.json \
     --output-dir environment/openfoam/results
   ```

### 12 工况自动收尾

正式的单频 `1.5 Hz`、每自由度双幅值实验可用持久监视器自动收尾：

```bash
python3 environment/openfoam/finish_cfd12.py \
  --cases-dir environment/openfoam/cases_cfd12_no_layers_level6_v1 \
  --config environment/openfoam/experiment_configs/cfd12_no_layers_level6_performance.json \
  --output-dir environment/openfoam/results_cfd12_no_layers_level6_v1 \
  --wait-seconds 30 \
  --runner-log environment/openfoam/cfd12_runner.log
```

`--runner-log` 可重复指定。脚本只在精确 `12` 个工况的 schema-v2 `.completed` 全部通过
`run_cases.py` 原始输出复验后才拟合；它也会观察明确指向该 cases 目录的 runner 进程。
未完成时若 runner 日志出现失败，或已观察到的全部 runner 提前退出，脚本会退出并原子写入
与输出目录同级的 `<output-dir>.failure.json`。成功路径在同级 staging 目录运行分析，默认用
`200` 次周期 bootstrap、`10000` 次被动性抽样并投影 `M_A` 为 PSD；只有 case 数、每 DOF
双幅值、秩 3、每工况 4 个完整周期、有限 `6×6` 矩阵、PSD、配置更新一致、被动性负分数为
零且末两周期比较均可用时，才把整个 staging 目录原子改名为最终输出目录。现有非匹配输出
目录不会被覆盖。

最终输出同时包含原始 `36` 项矩阵、可选物理投影矩阵、置信区间、设计秩/条件数、
逐通道 RMS、附加质量对称误差/特征值和阻尼被动性抽样，并生成可直接合并到本项目的：

```json
{
  "added_mass_diag": [["6x6"]],
  "linear_damping": [["6x6"]],
  "quadratic_damping": [["6x6"]]
}
```

运行时配置键 `added_mass_diag` 接受对角 `6` 向量或完整 `6×6` 矩阵。

## 已执行的 v2512 验证

本目录不是只做了字典渲染检查。部署阶段已用官方 OpenCFD v2512 二进制完成：

- 生成闭合测试外形并走完 `blockMesh`、`surfaceFeatureExtract`、完整
  `snappyHexMesh`（含 4 层边界层）和标准 `checkMesh`；
- 得到 `543093` 个单元，标准 `checkMesh` 报告 `Mesh OK`；
- 平移和转动算例各真实运行一个 `pimpleFoam` 动态网格时间步，均正常结束并写出
  流体力与力矩；
- 2 进程 MPI 平移动态网格时间步正常结束；
- 几何、建网格门禁、载荷解析与完整 108 系数合成恢复测试全部通过。

这些验证证明部署语法和数据链可以闭合。正式湿表面来自上述独立 staging 链路；发布矩阵
前仍需完成人工外形核对、网格收敛、时间步收敛和外域尺寸研究。

## 发布矩阵前的最低验证

- 至少三档网格、两档时间步和两档外域尺寸；
- 检查艇体层网格随最大运动仍保持正体积和合格非正交度；
- 比较每个频率/幅值单独拟合的系数漂移；
- 对 `M_A` 报告互易性误差和对称部分特征值，不隐藏原始非对称项；
- 对训练未使用的多轴运动做载荷时序验证；
- 检查 `nu^T tau_drag <= 0` 的被动性，不能仅靠逐元素系数正负判断；
- 压力载荷与黏性载荷分别检查，确认力矩已经移到同一 COM 参考点。
