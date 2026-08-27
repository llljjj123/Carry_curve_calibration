# 目标
本项目意在解决如下问题：

假设在时间$t = 0$，客户以价格$F_{0,T}$买入标的为$S$的股指期货，期货交割日为$T$，记此时此股指期货对应分红率为$q_{0,T}$，我们有$F_{0,T} = e^{\left(r - q_{0,T}\right)T}S_0$。在任意时间$\tau \in [0,T]$，假设此时分红率为$q_{\tau,T}$，客户拥有以价格$\max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)}S_\tau,F_{\tau,T}\right\}$平仓的权利，即客户在时间$\tau$时的盈亏为

$$
\begin{equation}
\max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)}S_\tau,F_{\tau,T}\right\} - F_{0,T}
\end{equation}
$$

进一步拆开上述方程，我们有：
$$
\begin{align}
(1) &= \max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)}S_\tau - F_{\tau,T},0\right\} + F_{\tau,T} - F_{0,T} \\
    &= \max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)} - e^{\left(r - q_{\tau,T}\right)\left(T-\tau\right)},0\right\}S_\tau + F_{\tau,T} - F_{0,T}
\end{align}
$$ 

其中$F_{\tau,T} - F_{0,T}$可以通过期货完全对冲，先暂不考虑。即我们只需考虑如何定价一个行权方式为美式的期权$\max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)} - e^{\left(r - q_{\tau,T}\right)\left(T-\tau\right)},0\right\}S_\tau$

我将其称为put因为这是对$e^{\left(r - q_{\tau,T}\right)\left(T-\tau\right)}$的put，对于分红率$q$本身则是call。

文章以下部分则将介绍此文件夹中具体包含了哪些文件，以及我们如何处理上述期权定价。

# 结构
本项目包含5个文件夹，其中`im_ou_carry`, `im_2factor_ou_carry`和`im_corr_ou_1factor`是分红率（即$q$，这里被叫做`carry rate`）模型；`carry_put_pricing`是期权定价模型，其中对应的分红率模型为`im_2factor_ou_carry`；`Demo`则是一个简单的场景演示，里面包含了从拟合分红率到期权定价的全过程。

同时项目内还包含一个`summary_for_AI.md`文件，此文件可以帮助AI更好地理解项目结构和内容。在使用AI的时候直接把此文件交给它阅读即可。

# 分红率模型
因为能观察到分红率$q$往往具有较强的均值回归性，所以作为模型初探，我选择了vasicek作为起点。同时观察到有时$q$和价格$S$具有相关性，即$S$上涨$q$也变大，$S$下降$q$变小，所以同时提出一个$q$和$S$具有相关性的模型。

## 单$\kappa$ vasicek模型
此模型对应`im_ou_carry`（因为gpt使用了OU process这个叫法，所以我也命名为ou）,在这个模型中，我们假设instantaneous分红率为$c_t$，依据vasicek我们有

$$
dc_t=\kappa(\theta-c_t)dt+\eta dW_t.
$$

对$q_{t,T}$和$c_t$我们可得出如下：

约定
$$
B(t) = \frac{1-e^{-\kappa(t)}}{\kappa}\\
C(t) = \int_0^t B(s)^2 ds
$$
根据$F_{t,T} = e^{\left(r - q_{t,T}\right)\left(T-t\right)}St = \mathbb{E}^*_t\left(S_T\right)$，我们可以得出市场隐含分红率$q_{t,T}$和instantaneous分红率$c_t$的关系：

$$
q_{t,T}=\theta+\frac{B\left(T-t\right)}{(T-t)}(c_t-\theta) - \frac{\eta^2}{2\left(T-t\right)}C\left(T-t\right)
$$

忽略掉convexity项$\frac{\eta^2}{2\left(T-t\right)}C\left(T-t\right)$，加入市场噪音，我们用于Kalman filter的公式为

$$
q_{t,T}=\theta+\frac{1-e^{-\kappa(T-t)}}{\kappa(T-t)}(c_t-\theta)+\varepsilon_{t,T},
$$

其中$\varepsilon_{t,T}$即代表市场噪音。

拟合时，我们每天同时使用当日所有可用的IM期货合约，通过Kalman filter从整条期限曲线中过滤$c_t$；状态在两个观测日之间的变化使用vasicek过程的转移公式，并按照实际交易日间隔处理。随后通过最大似然估计同时求解$\kappa,\theta,\eta$以及观测误差波动率$\sigma_\varepsilon$。优化使用多个初始值，以降低局部最优解对结果的影响。

此模型的拟合结果可以较好描述整体分红率水平的均值回归，但它生成的期限曲线只能单调地向$\theta$靠拢，因此无法拟合市场中实际出现的U形或倒U形曲线。样本中约27.75%的日期被识别为存在U形或倒U形特征，这也是下面引入快慢$\kappa$的主要原因。

## 快慢$\kappa$ vasicek模型
此模型对应`im_2factor_ou_carry`，因为单$\kappa$ vasicek对于U形或者倒U形的分红率曲线无法良好拟合，所以选用两$\kappa$，即一快一慢$\kappa$模型来拟合

快慢$\kappa$模型将即期分红率拆成一个长期水平和两个均值为0的状态：

$$
c_t=\theta+x_{s,t}+x_{f,t},
$$

$$
dx_{j,t}=-\kappa_jx_{j,t}dt+\eta_jdW_{j,t},\qquad 0<\kappa_s<\kappa_f.
$$

其中$x_s$是慢因子，主要影响整条曲线和较长期限；$x_f$是快因子，主要影响曲线前端。对于剩余期限$T-t$，同样忽略convexity项，模型给出的平均隐含分红率为

$$
q_{t,T}=\theta+B(\kappa_s,T-t)x_{s,t}+B(\kappa_f,T-t)x_{f,t}+\varepsilon_{t,T},
$$

其中$B(\kappa,\tau)=(1-e^{-\kappa\tau})/(\kappa\tau)$。当快慢因子的符号相反时，两种衰减速度的叠加可以产生U形或倒U形，因此比单$\kappa$模型具有更强的期限结构拟合能力。当前版本假设快慢因子的随机冲击相互独立，并对所有期限使用同一个观测误差波动率。

拟合方法仍然是对每日所有合约进行联合Kalman filter和多初始值最大似然估计，但状态维度由1维变为2维。模型通过参数变换强制满足$\kappa_s>0$以及$\kappa_f>\kappa_s$，并使用精确的非等间隔OU转移。主项目还会在完全相同的数据、时间口径和样本切分上重新估计单$\kappa$模型，以保证模型比较具有可比性。

双因子模型仍未完全消除残差的自相关和波动聚集。此外，因快因子衰减速度极快，若当最近的可交易合约仍然较远时，对快因子可能识别较弱。约21.49%的历史日期被标记为即期状态可观测性较弱（即最近到期日大于21个交易日，或过滤后的即期carry标准差高于0.04）。滚动窗口中也存在$\eta_f$触及上限的情况，因此快因子的短端解释和短期期权应用都应结合参数稳定性诊断，而不能只看单次最优估计。

## 单$\kappa$相关性模型
此模型对应`im_corr_ou_1factor`，我们加入了分红率和价格的相关性$\rho$，并验证了二者是否真具有相关性。这一部分均为AI生成，并得出相关性$\rho$不显著的结论，我对此稍微存疑，下一步可进一步改进calibration方法，例如先固定$\rho$进行初次模拟以确定一个其他参数的初始值，再放开$\rho$进行二次模拟，以提高拟合结果。

相关性模型在单$\kappa$ vasicek模型的基础上令现货收益冲击与分红率冲击相关：

$$
\frac{dS_t}{S_t}=(r-c_t)dt+\sigma dW_t^S,
$$

$$
dc_t=\kappa(\theta-c_t)dt+\eta dW_t^c,
\qquad dW_t^S dW_t^c=\rho dt.
$$

这一版本先将中证1000年化波动率固定为$\sigma=25\%$。与前面的单因子模型不同，这里使用随机分红率下的精确期货定价公式：

$$
\log\frac{F(t,T)}{S_t}
=(r-\theta)\tau-(c_t-\theta)B(\tau)
+\frac{1}{2}\eta^2C(\tau)-\rho\sigma\eta D(\tau).
$$

其中$B,C,D$分别来自OU过程在$[t,T]$上的积分及协方差项。$B,C$已在上方定义，$D(t) = \int_0^tB(s)ds$

为了避免错误地解释相关性，项目分别拟合了5个模型：原始单因子曲线模型、精确定价且$\rho=0$的曲线模型、自由估计$\rho$的曲线模型、$\rho=0$的曲线和收益联合模型，以及自由估计$\rho$的联合模型。原始单因子模型没有包含精确公式中的凸性修正，因此不能把它直接当成相关模型在$\rho=0$时的受限版本；似然比检验只在两个精确曲线模型之间以及两个精确联合模型之间进行。

结果显示，目前的数据并不支持$\rho$显著偏离0。只使用期货曲线时，点估计为$\rho=-0.6420$，但近似标准误高达2.1303，profile likelihood在测试的$[-0.9,0.9]$范围内几乎是平的；$\rho=0$的似然比检验$p$值为0.7363。因此这个较大的负数主要反映参数无法被曲线单独识别，而不是强负相关的经济证据。

使用期货曲线和现货收益的联合模型后，估计变为$\rho=-0.03277$，近似标准误为0.03605，95%的profile likelihood区间约为$[-0.1029,0.0378]$，仍然包含0；$\rho=0$的似然比检验$p$值为0.3639。联合模型在不同滚动样本中的$\rho$还出现了明显的符号变化，样本外表现也没有优于$\rho=0$模型。综合来看，联合数据可以排除非常大的相关性，却不能拒绝零相关；在当前固定25%现货波动率和单因子设定下，没有充分证据表明加入非零$\rho$能够改善模型。

# 定价模型
此部分对应`carry_put_pricing`，我们使用快慢$\kappa$ vasicek模型作为分红率模型，并在此模块中提出对上述期权的定价方法。

令

$$
I_{t,T}=\int_t^T c_u du,
$$

由于OU过程的积分服从条件正态分布，未来的期货价格可以写成

$$
\frac{F_{t,T}}{S_t}
=e^{r(T-t)}E_t^Q[e^{-I_{t,T}}]
=\exp\left(r(T-t)-E_t^Q[I_{t,T}]
+\frac{1}{2}\operatorname{Var}_t^Q(I_{t,T})\right).
$$

在数值方法上注意到赔付方程
$$
\max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)} - e^{\left(r - q_{\tau,T}\right)\left(T-\tau\right)},0\right\}S_\tau
$$

存在类似如下特点，即标的价格$S$与$\max(...)$为齐一次相乘

$$
V(S,...)=S\,v(...).
$$

故做如下处理，令

$$
P_\tau = \max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)} - e^{\left(r - q_{\tau,T}\right)\left(T-\tau\right)},0\right\}S_\tau\\

v_\tau = \frac{P_\tau}{S_\tau} = \max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)} - e^{\left(r - q_{\tau,T}\right)\left(T-\tau\right)},0\right\}
$$

即定价目标有原$P_\tau$转为$v_\tau$，此举可以将$S$从需要离散化的状态变量中移除。

程序从到期日向前进行动态规划：先根据OU过程计算一个交易日的精确状态转移和积分矩，再通过Gaussian exponential tilting处理随机分红率带来的权重，使用Gauss-Hermite求积计算条件期望，并通过双线性插值把求积点映射到状态网格上；在每个交易日比较立即行权价值与继续持有价值并取较大者。由于快慢两个因子相互独立，二维正态求积可以拆成连续的两个一维变换，从而显著减少计算量。

程序允许每个交易日行权，因此严格来说是对连续行权美式期权的逐日Bermudan近似。


# 演示
对应`Demo`，演示了一种最简单的假设业务情况。即客户于一个初始给定日期买入一笔期货（为简单暂定为中金所期货，代码中使用IM2612）和一个上述美式看跌，两者到期日一致。同时在程序中给定一个用于拟合快慢$\kappa$参数的时间窗口，程序会自动开始拟合，并汇报拟合结果。最后程序会用拟合结果对上述美式看跌进行定价，此价格即为我们（卖方）在入场交易时应当对客户额外收取的指数点（如定价为$50$，则对应一笔IM，应当收取$50\times 200 = 10000$元）


