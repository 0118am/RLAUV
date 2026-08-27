# 完整响应矩阵拟合器

分析器从给出的 24 个工况目录直接读取 `case.json` 和原始
`force.dat/moment.dat`。输出坐标是 moving-COM body FLU，顺序为
`[u,v,w,p,q,r]` / `[X,Y,Z,K,M,N]`。

- 稳态正负配对的半差拟合平移 `D_L,D_Q`；全部六个 wrench 响应共同形成对应激励列。
- 六个低幅值附加质量工况做半周期奇投影；其中三个平移结果用于平移 `M_A`，三个
  转动结果用于检查幅值依赖。
- 每个转动轴的两档转速工况联合回归 `[-nu_dot,-nu,-|nu|nu]`，同时拟合六个响应通道的
  `M_A,D_L,D_Q`，不再固定低幅值转动附加质量。
- 依据 body-FLU 左右镜像对称，只清零偶/奇块之间物理禁止的项；两个块内允许的非对角项
  保留。独立拟合的 `M_A` 列按互易性取对称平均，不对 `D_L,D_Q` 做裁剪或被动性投影。

输出保留每个 wrench 通道的系数、NRMSE、最后周期/稳态窗口变化、横轴载荷比例、附加质量
互易误差和线性阻尼对称部分特征值作为诊断量，不设置通过/拒绝阈值。比较横轴载荷前先用
艇长把力矩换算为 `M/L`，避免直接混用 N 与 N·m。

运行：

```bash
python3 -m environment.openfoam.analysis \
  --cases-root environment/openfoam/cases/current \
  --config environment/openfoam/config.json
```
