## 第四部分 策略学习：从策略梯度、Actor-Criti到TRPO、PPO算法

## 4.1 策略梯度与其突出问题：采样效率低下

本节推导的核心内容参考自Easy RL教程等资料(但修正了原教程上部分不太准确的描述，且为让初学者更好懂，补充了大量的解释说明和心得理解，倪老师则帮拆解了部分公式)。

另，都说多一个公式则少一个读者，本文要打破这点，虽然本节推导很多，但每一步推导都有介绍到，不会省略任何一步推导，故不用担心看不懂(对本文任何内容有任何问题，都欢迎随时留言评论)。

### 4.1.1 什么是策略梯度和梯度计算/更新的流程

策略梯度的核心算法思想是：
+   参数为$`\theta`$的策略$`\pi_{\theta }`$接受状态，输出动作概率分布，在动作概率分布中采样动作，执行动作(形成运动轨迹$`\tau`$)，得到奖励$`r`$，跳到下一个状态
+   在这样的步骤下，可以使用策略$`\pi`$收集一批样本，然后使用梯度下降算法学习这些样本，不过当策略$`\pi`$的参数更新后，这些样本不能继续被使用，还要重新使用策略$`\pi`$与环境互动收集数据

比如REINFORCE算法便是常见的策略梯度算法，类似下图所示(下图以及本节大部分配图/公式均来自easy RL教程)

![](./assets/images/RL_simple_primer/9dcf9cefeb844a5a93b2b6fd38bf5a80.png)

接下来，详细阐述。首先，我们已经知道了策略函数可以如此表示：$`a = \pi _{\theta }(s)`$

其中，$`\pi _{\theta}`$可以理解为一个我们所熟知的神经网络
+   当你对神经网络有所了解的话，你一定知道通过梯度下降求解损失函数的极小值（忘了的，可以复习下：首先通过正向传播产生拟合值，与标签值做“差”计算，产生误差值，然后对误差值求和产生损失函数，最后对损失函数用梯度下降法求极小值，而优化的对象就是神经网络的参数$`\theta`$

+   类比到πθ这个问题上，现在是正向传播产生动作，然后动作在环境中产生奖励值，通过奖励值求和产生评价函数，此时可以针对评价函数做梯度上升（gradient ascent），毕竟能求极小值，便能求极大值，正如误差能最小化，奖励/得分就能最大化

如何评价策略的好坏呢？

假设机器人在策略$`\pi_{\theta }`$的决策下，形成如下的运动轨迹(类似你玩三国争霸时，你控制角色在各种不同的游戏画面/场景/状态下作出一系列动作，而当完成了系统布置的某个任务时则会得到系统给的奖励，如此，运动轨迹用$`\tau`$表示，从而$`\tau`$表示为一个状态$`s`$、动作$`a`$、奖励值$`r`$不断迁移的过程)

$`\tau = (s_{1},a_{1},r_{1},s_{2},a_{2},r_{2},...,s_{t},a_{t},r_{t})`$

> 可能有读者注意到了，既然奖励是延后的，$`s_t$,$a_t`$后的奖励怎么用$`r_t`$而非$`r_{t+1}`$呢，事实上，sutton RL书上用$`S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,\cdots,S_t,A_t,R_{t+1}`$表示整条轨迹，其实这样更规范，但考虑到不影响大局和下文的推导，本笔记则暂且不细究了

给定智能体或演员的策略参数$`\theta`$，可以计算某一条轨迹$`\tau`$发生的概率为『轨迹$`\tau`$来源于在特定的环境状态下采取特定动作的序列，而特定的状态、特定的动作又分别采样自智能体的动作概率分布$`p_{\theta }(a_{t}|s_{t})`$、状态的转换概率分布$`p(s_{t+1}|s_t,a_t)`$

$`\begin{aligned} p_{\theta}(\tau) &=p\left(s_{1}\right) p_{\theta}\left(a_{1} | s_{1}\right) p\left(s_{2} | s_{1}, a_{1}\right) p_{\theta}\left(a_{2} | s_{2}\right) p\left(s_{3} | s_{2}, a_{2}\right) \cdots \\ &=p\left(s_{1}\right) \prod_{t=1}^{T} p_{\theta}\left(a_{t} | s_{t}\right) p\left(s_{t+1} | s_{t}, a_{t}\right) \end{aligned}`$

其中，有的资料也会把$`p_{\theta }(a_{t}|s_{t})`$写成为$`\pi _{\theta }(a_{t}|s_{t})`$，但由于毕竟是概率，所以更多资料还是写为$`p_{\theta }(a_{t}|s_{t})`$

如何评价策略呢？这个策略评价函数为方便理解也可以称之为策略价值函数，就像上文的状态价值函数、动作价值函数，说白了，评估策略(包括状态、动作)的价值，就是看其因此得到的期望奖励

故考虑到期望的定义，由于每一个轨迹$`\tau`$ 都有其对应的发生概率，**对所有$`\tau`$出现的概率与对应的奖励进行加权最后求和**，即可得期望值：

$`\bar{R}_{\theta}=\sum_{\tau} R(\tau) p_{\theta}(\tau)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}[R(\tau)]`$

上述整个过程如下图所示

![](./assets/images/RL_simple_primer/756685e2f07b494b99bc97f4ce0f4bf9.png)

通过上文已经知道，想让奖励越大越好，可以使用梯度上升来最大化期望奖励。而要进行梯度上升，先要计算期望奖励$`\bar{R}_{\theta}`$的梯度。

考虑对$`\bar{R}_{\theta}`$做梯度运算『再次提醒，忘了什么是梯度的，可以通过[一文通透优化算法：从梯度下降、SGD到牛顿法、共轭梯度(23修订版)](https://blog.csdn.net/v_JULY_v/article/details/81350035 "一文通透优化算法：从梯度下降、SGD到牛顿法、共轭梯度(23修订版)")复习下』

$`\nabla \bar{R}_{\theta}=\sum_{\tau}{R}(\tau )\nabla \mathrm{p}_{\theta}(\tau )`$

其中，只有$`p_{\theta}(\tau)`$与$`\theta`$有关。再考虑到$`\nabla f(x)=f(x)\nabla \log f(x)`$，可得

$`\frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)}= \nabla\log p_{\theta}(\tau)`$

从而进一步转化，可得$`\begin{aligned} \nabla \bar{R}_{\theta}&=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[R(\tau) \nabla \log p_{\theta}(\tau)\right] \end{aligned}`$，表示期望的梯度等于对数概率梯度的期望乘以原始函数

> Em，怎么来的？别急，具体推导是
>
> $`\begin{aligned} \nabla \bar{R}_{\theta}&=\sum_{\tau} R(\tau) \nabla p_{\theta}(\tau)\\&=\sum_{\tau} R(\tau) p_{\theta}(\tau) \frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)} \\&= \sum_{\tau} R(\tau) p_{\theta}(\tau) \nabla \log p_{\theta}(\tau) \\ &=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[R(\tau) \nabla \log p_{\theta}(\tau)\right] \end{aligned}`$
>
> 上述推导总共4个等式3个步骤，其中，第一步 先分母分子都乘以一个$`p_{\theta}(\tau)`$，第二步 把$`\frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)}= \nabla \log p_{\theta}(\tau)`$代入计算，第三步 根据期望的定义$`E[X] = \sum_{i}^{}p_ix_i`$做个简单转换，此处的$`X`$就是$`R(\tau )`$
>
> 此外，本文一读者在23年2.24日的留言说，还想了解$`\nabla f(x)=f(x)\nabla \log f(x)`$是怎么推导而来的，这个式子可以通过如下推导得到
>
> 首先，对函数$`f(x)`$取对数得：
>
> $`\log f(x)`$
>
> 对上式求导数得：
>
> $`\frac{d}{dx}\log f(x) = \frac{1}{f(x)}\frac{d}{dx}`$
>
> 将等式两边同乘以$`f(x)`$，得到：
>
> $`f(x) \frac{d}{dx} \log f(x) = \frac{d}{dx}`$
>
> 这个等式表明，我们可以用$`\nabla \log f(x)`$来表示$`\nabla f(x)`$，即：
>
> $`\nabla f(x)=f(x)\nabla \log f(x)`$

然不巧的是，期望值$`\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[R(\tau) \nabla \log p_{\theta}(\tau)\right]`$无法计算，按照蒙特卡洛方法近似求期望的原则，可以采样$`N`$条轨迹$`\tau`$并计算每一条轨迹的值，再把每一条轨迹的值加起来除以$`N`$取平均，即($`\tau^{n}`$上标$`n`$代表第$`n`$条轨迹，而$`a_{t}^{n}`$、$`s_{t}^{n}`$则分别代表第$`n`$条轨迹里时刻$`t`$的动作、状态)

$`\begin{aligned} \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[R(\tau) \nabla \log p_{\theta}(\tau)\right] &\approx \frac{1}{N} \sum_{n=1}^{N} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(\tau^{n}\right) \\ &=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(a_{t}^{n} \mid s_{t}^{n}\right) \end{aligned}`$

> 任何必要的中间推导步骤咱不能省，大部分文章基本都是一笔带过，但本文为照顾初学者甚至更初级的初学者，$`\nabla \log p_{\theta}(\tau)`$中间的推导过程还是要尽可能逐一说明下：
>
> 1.  首先，通过上文中关于某一条轨迹$`\tau`$发生概率的定义，可得
>
>     $`p_\theta (\tau ) = p(s_{1}) \prod_{t=1}^{T_{n}}p(s_{t+1}|s_t,a_t)p_{\theta }(a_{t}|s_{t})`$
>
> 2.  然后两边都取对数，可得
>
>     $`logp_\theta (\tau ) = logp(s_1)\prod_{t=1}^{T_{n}} p(s_{t+1}|s_t,a_t)p_{\theta }(a_{t}|s_{t})`$
>
>     由于乘积的对数等于各分量的对数之和，故可得
>
>     $`logp_\theta (\tau ) = logp(s_1) + \sum_{t=1}^{T_n}(logp(s_{t+1}|s_t,a_t) + logp_{\theta }(a_{t}|s_{t}))`$
>
> 3.  接下来，取梯度可得
>
>     $`\begin{aligned} \nabla \log p_{\theta}(\tau) &= \nabla \left(\log p(s_1)+ \sum_{t=1}^{T_n}\log p(s_{t+1}|s_t,a_t) + \sum_{t=1}^{T_n}\log p_{\theta}(a_t|s_t) \right) \\ &= \nabla \log p(s_1)+ \nabla \sum_{t=1}^{T_n}\log p(s_{t+1}|s_t,a_t) + \nabla \sum_{t=1}^{T_n}\log p_{\theta}(a_t|s_t) \\ &=\nabla \sum_{t=1}^{T_n}\log p_{\theta}(a_t|s_t)\\ &=\sum_{t=1}^{T_n} \nabla\log p_{\theta}(a_t|s_t) \end{aligned}`$
>
>     上述过程总共4个等式，在从第2个等式到第3个等式的过程中，之所以消掉了
>
>     $`\nabla \log p(s_1)+\nabla \sum_{t=1}^{T_n}{\log}p(s_{t+1}|s_t,a_t)`$
>
>     是因为其与$`\theta`$无关(环境状态不依赖于策略)，其对$`\theta`$的梯度为0。

完美！这就是所谓的**策略梯度定理**，我们可以直观地理解该公式

$`\nabla \bar{R}_{\theta}=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)`$

1.  即在采样到的数据里面，采样到在某一个状态$`s_t`$要执行某一个动作$`a_t`$，$`(s_t,a_t)`$是在整个轨迹$`\tau`$的里面的某一个状态和动作的对
2.  为了最大化奖励，假设在$`s_t`$执行$`a_t`$，最后发现$`\tau`$的奖励是正的，就要增加在$`s_t`$ 执行$`a_t`$的概率。反之，如果在$`s_t`$执行$`a_t`$会导致$`\tau`$的奖励变成负的， 就要减少在$`s_t`$执行$`a_t`$的概率
3.  最后，用梯度上升来更新参数，原来有一个参数$`\theta`$，把$`\theta`$加上梯度$`\nabla \bar{R}_{\theta}`$，当然要有一个学习率$`\eta`$（类似步长、距离的含义），学习率的调整可用 Adam、RMSProp等方法调整，即

$`\theta \leftarrow \theta+\eta \nabla \bar{R}_{\theta}`$

> 有一点值得说明的是...，为了提高可读性，还是举个例子来说明吧。
>
> 比如到80/90后上大学时喜欢玩的另一个游戏CF(即cross fire，10多年前我在东华理工的时候也经常玩这个，另一个是DNF)，虽然玩的是同一个主题比如沙漠战场，但你每场的发挥是不一样的，即便玩到同一个地方(比如A区埋雷的地方)，你也可能会控制角色用不同的策略做出不同的动作，比如
>+   在第一场游戏里面，我们在状态$`s_1`$采取动作 $`s_1`$，在状态$`s_2`$采取动作 $`a_2`$。且你在同样的状态$`s_1`$下， 不是每次都会采取动作$`a_1`$的，所以我们要记录，在状态 $`s^1_1`$ 采取 $`a^1_1`$、在状态 $`s^1_2`$采取$`a^1_1`$等，整场游戏结束以后，得到的奖励是 $`R(\tau^1)`$
>+   在第二场游戏里面，在状态$`s^2_1`$采取$`a^2_1`$，在状态 $`s^2_2`$采取$`a^2_2`$，采样到的就是$`\tau^2`$，得到的奖励是$`R(\tau^2)`$
> 这时就可以把采样到的数据用梯度计算公式把梯度算出来
>
> 1.  也就是把每一个$`s`$与$`a`$的对拿进来，计算在某一个状态下采取某一个动作的对数概率$`\log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)`$，对这个概率取梯度$`\nabla \log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)`$
> 2.  然后在梯度前面乘一个权重$`R\left(\tau^{n}\right)`$，权重就是这场游戏的奖励，这也是和一般分类问题的区别所在
>
>     ![](./assets/images/RL_simple_primer/bad281ae107d49d3adf8aa2d012a08c1.png)
>
> 3.  计算出梯度后，就可以通过$`\theta \leftarrow \theta+\eta \nabla \bar{R}_{\theta}`$更新模型了

### 4.1.2 避免采样的数据仅能用一次：重要性采样(为采样q解决p从而增加重要性权重)

然而策略梯度有个问题，在于$`\mathbb{E}_{\tau \sim p_{\theta}(\tau)}`$是对策略$`{\pi _{\theta}}`$采样的轨迹$`\tau`$求期望。一旦更新了参数，从$`\theta`$变成$`\theta'`$，在对应状态s下采取动作的概率$`p_\theta(\tau)`$就不对了，之前采样的数据也不能用了。

换言之，策略梯度是一个会花很多时间来采样数据的算法，其大多数时间都在采样数据。智能体与环境交互以后，接下来就要更新参数，我们只能更新参数一次，然后就要重新采样数据， 才能再次更新参数。

![](./assets/images/RL_simple_primer/0dbb490df24a8ddc53d81df6b09c9c76.png)

这显然是非常花时间的，怎么解决这个问题呢？为避免采样到的数据只能使用一次的问题，还记得上文介绍过的重要性采样否，使得

> 1.  可以用另外一个策略$`\pi_{\theta'}`$与环境交互，用$`\theta'`$采样到的数据去训练$`\theta`$
> 2.  假设我们可以用$`\theta'`$采样到的数据去训练$`\theta`$，我们可以多次使用$`\theta'`$采样到的数据，可以多次执行梯度上升，可以多次更新参数$`\theta`$， 都只需要用$`\theta'`$采样到的同一批数据

故基于重要性采样的原则，我们可以用另外一个策略$`\pi _{\theta^{'}}`$，与环境做互动采样数据来训练$`\theta`$，从而间接计算$`R(\tau) \nabla \log p_{\theta}(\tau)`$，而当我们转用$`\theta'`$去采样数据训练$`\theta`$后

1.  只需在$`R(\tau) \nabla \log p_{\theta}(\tau)`$的基础上补上一个重要性权重：$`\frac{p_{\theta}(\tau)}{p_{\theta^{\prime}}(\tau)}`$，这个**重要性权重**针对某一个轨迹$`\tau`$用$`\theta`$算出来的概率除以这个轨迹$`\tau`$用$`\theta^{'}`$算出来的概率
2.  注意，上面例子中的$`p`$/$`q`$与此处的$`p_{\theta}(\tau)/p_{\theta^{\prime}}(\tau)`$没有任何联系，前者只是为了说明重要性权重的两个普通的分布而已

最终加上重要性权重之后，可得

$`\nabla \bar{R}_{\theta}=\mathbb{E}_{\tau \sim p_{\theta^{\prime}(\tau)}}\left[\frac{p_{\theta}(\tau)}{p_{\theta^{\prime}}(\tau)} R(\tau) \nabla \log p_{\theta}(\tau)\right]`$

怎么来的？完整推导如下

$`\begin{aligned}\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[R(\tau) \nabla \log p_{\theta}(\tau)\right] &= \sum_{\tau} \left[R(\tau) \nabla \log p_{\theta}(\tau)\right]p_{\theta}(\tau) \\ &= \sum_{\tau} \left[\frac{p_{\theta}(\tau)}{p_{\theta}^\prime(\tau)}R(\tau) \nabla \log p_{\theta}(\tau)\right]p_{\theta}^\prime(\tau) \\ &= \mathbb{E}_{\tau \sim p_{\theta^{\prime}(\tau)}}\left[\frac{p_{\theta}(\tau)}{p_{\theta^{\prime}}(\tau)} R(\tau) \nabla \log p_{\theta}(\tau)\right] \\ & = \nabla \bar{R}_{\theta}\end{aligned}`$

## 4.2 优势演员-评论家算法(Advantage Actor-Criti)：为避免奖励总为正增加基线

梯度的计算好像差不多了？但实际在做策略梯度的时候，并不是给整个轨迹$`\tau`$都一样的分数，而是每一个状态-动作的对会分开来计算，但通过蒙特卡洛方法进行随机抽样的时候，可能会出问题，比如在采样一条轨迹时可能会出现

+   问题1：所有动作均为正奖励
+   问题2：出现**比较大的方差**(另，重要性采样时，采样的分布与当前分布之间也可能会出现比较大的方差，具体下一节详述)

对于第一个问题，举个例子，比如在某一一个状态，可以执行的动作有a、b、c，但我们可能只采样到动作b或者只采样到动作c，没有采样到动作a

1.  但不管采样情况如何，现在所有动作的奖励都是正的，所以采取a、b、c的概率都应该要提高
2.  可实际最终b、c的概率按预期提高了，但因为a没有被采样到，所以a的概率反而下降了
3.  然而问题是a不一定是一个不好的动作，它只是没有被采样到

![](./assets/images/RL_simple_primer/bc1f965ef46e4ea693bb1950fd76d7e8.png)

为了解决奖励总是正的的问题，也为避免方差过大，需要在之前梯度计算的公式基础上加一个基准线$`b`$『此$`b`$指的baseline，非上面例子中的$`b`$，这个所谓的基准线$`b`$可以是任意函数，只要不依赖于动作$`a`$即可』

$`\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}}\left(R\left(\tau^{n}\right)-b\right) \nabla \log p_{\theta}\left(a_{t}^{n} \mid s_{t}^{n}\right)`$

上面说$`b`$可以是任意函数，这个“任意”吧，对初学者而言可能跟没说一样，所以$`b`$到底该如何取值呢
+   $`b`$有一种选择是使用轨迹上的奖励均值，即  $`b=\frac{1}{T}\sum_{t=1}^{T}R_t(\tau)`$    
    从而使得$`R(\tau)−b`$有正有负  
    当$`R(\tau)`$大于平均值$`b`$时，则$`R(\tau)−b`$为正，则增加该动作的概率  
    当$`R(\tau)`$小于平均值$`b`$时，则$`R(\tau)−b`$为负，则降低该动作的概率  
    如此，对于每条轨迹，平均而言，较好的50%的动作将得到奖励，避免所有奖励均为正或均为负，同时，也减少估计方差  
+   $`b`$还可以是状态价值函数$`V_{\pi}(st)`$  
    可曾还记得2.1节介绍过的所谓Actor-Criti算法(一般被翻译为演员-评论家算法)  
    Actor学习参数化的策略即策略函数，Criti学习值函数用来评估状态-动作对，然后根据评估结果改进或更新策略

    当然，Actor-Criti本质上是属于基于策略的算法，毕竟算法的目标是优化一个带参数的策略(实际用到PPO算法时，会计算一个策略损失)，只是会额外学习价值函数(相应的，运用PPO算法时，也会计算一个价值损失)，从而帮助策略函数更好的学习，而学习优势函数的演员-评论家算法被称为优势演员-评论家(Advantage Actor-Criti，简称A2C)算法  

而这个$`R(\tau)-b`$一般被定义为优势函数$`A^{\theta}(s_t,a_t)`$，有几点值得注意:  

1.  在考虑到评估动作的价值，就看其因此得到的期望奖励，故一般有$`A_\pi (s,a) = Q_\pi (s,a) - V_\pi (s)`$，此举意味着在选择一个动作时，根据该动作相对于特定状态下其他可用动作的执行情况来选择，而不是根据该动作的绝对值(由$`Q`$函数估计)  
且通常我们**只学习$`V_\pi (s)`$「比如通过时序差分法估计」，然后通过$`V_\pi (s)`$与奖励的结合来估计$`Q_\pi`$,即$`Q_{\pi}=R+\gamma V\pi (st+1)`$
，从而可得**  
$`A\pi (s,a)=Q\pi (s,a)−V\pi (s)=R+\gamma V\pi (st+1)−V\pi (s)`$  

2.  **总之，$`A^{\theta }(s_{t},a_{t})`$要估测的是在状态$`s_{t}`$采取动作$`a_{t}`$是好的还是不好的：即  
$`→`$如果$`A^{\theta }(s_{t},a_{t})`$是正的(即大于0)，意味着在状态 $`s_{t}`$ 采取动作 $`a_{t} `$获得的回报比在状态 $`s_{t}`$采取任何其他可能的动作获得的回报都要好，就要增加概率；  
$`→`$如果$`A^{\theta }(s_{t},a_{t})`$是负的(即小于0)，意味着在状态 $`s_{t}`$ 采取动作 $`a_{t}`$ 得到的回报比其他动作的平均回报要差，要减少概率**  

3.  最终在更新梯度的时候，如下式所示『我们用演员$`\theta`$去采样出$`s_{t}`$跟$`a_{t}`$，采样出状态跟动作的对$`(s_{t},a_{t})`$，计算这个状态跟动作对的优势$`A^{\theta }(s_{t},a_{t})`$』
```math
\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta}}\left[A^{\theta}\left(s_{t}, a_{t}\right) \nabla \log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)\right]
```


进一步，由于$`A^{\theta}(s_t,a_t)`$是演员$`\theta`$与环境交互的时候计算出来的，基于重要性采样的原则，当从$`\theta`$换到$`\theta'`$的时候，就需要在
```math
\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta}}\left[A^{\theta}\left(s_{t}, a_{t}\right) \nabla \log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)\right]
```

基础上，$`A^{\theta}(s_t,a_t)`$变换成$`A^{\theta'}(s_t,a_t)`$，一变换便得加个重要性权重(即把$`s_{t}`$、$`a_{t}`$用$`\theta`$采样出来的概率除掉$`s_{t}`$、$`a_{t}`$用$`\theta^{'}`$采样出来的概率)，公式如下『Easy RL纸书第1版上把下面公式中的$`A^{\theta'}(s_t,a_t)`$写成了$`A^{\theta}(s_t,a_t)`$』
```math
\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_{\theta}\left(s_{t}, a_{t}\right)}{p_{\theta^{\prime}}\left(s_{t}, a_{t}\right)} A^{\theta'}\left(s_{t}, a_{t}\right) \nabla \log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)\right]
```

接下来，我们可以拆解$`p_{\theta}(s_{t}, a_{t})`$和$`p_{\theta'}\left(s_{t}, a_{t}\right)`$，即
```math
\begin{aligned} p_{\theta}\left(s_{t}, a_{t}\right)&=p_{\theta}\left(a_{t}|s_{t}\right) p_{\theta}(s_t) \\ p_{\theta'}\left(s_{t}, a_{t}\right)&=p_{\theta'}\left(a_{t}|s_{t}\right) p_{\theta'}(s_t) \end{aligned}
```

于是可得公式
```math
\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta^{\prime}}\left(a_{t} | s_{t}\right)} \frac{p_{\theta}\left(s_{t}\right)}{p_{\theta^{\prime}}\left(s_{t}\right)} A^{\theta^{\prime}}\left(s_{t}, a_{t}\right) \nabla \log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)\right]
```

这里需要做一件事情，假设模型是$`\theta`$的时候，我们看到$`s_{t}`$的概率，跟模型是$`\theta^{'}`$的时候，看到$`s_{t}`$的概率是差不多的，即$`p_{\theta}(s_t)=p_{\theta'}(s_t)`$。

> 为什么可以这样假设呢？一种直观的解释就是$`p_{\theta}(s_t)`$很难算，这一项有一个参数$`\theta`$，需要拿$`\theta`$去跟环境做互动，算$`s_{t}`$出现的概率。 尤其是如果输入是图片的话，同样的$`s_{t}`$根本就不会出现第二次。我们根本没有办法估这一项，所以就直接无视这个问题。
>
> 但是$`p_{\theta}(a_t|s_t)`$是很好算，我们有$`\theta`$这个参数，它就是个网络。我们就把$`s_{t}`$带进去，$`s_{t}`$就是游戏画面。 我们有个策略的网络，输入状态$`s_{t}`$，它会输出每一个$`a_{t}`$的概率。所以，我们只要知道$`\theta`$和$`\theta'`$的参数就可以算$`\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta^{\prime}}\left(a_{t} | s_{t}\right)}`$。

所以，实际上在更新参数的时候，我们就是按照下式来更新参数：

$`\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta^{\prime}}\left(a_{t} | s_{t}\right)} A^{\theta^{\prime}}\left(s_{t}, a_{t}\right) \nabla \log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)\right]`$

从而最终可以从梯度$`\nabla f(x)=f(x) \nabla \log f(x)`$来反推目标函数，当使用重要性采样的时候，要去优化的目标函数如下式所示，把它记$`J^{\theta^{\prime}}(\theta)`$

$`J^{\theta^{\prime}}(\theta)=\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta^{\prime}}\left(a_{t} | s_{t}\right)} A^{\theta^{\prime}}\left(s_{t}, a_{t}\right)\right]`$

终于大功告成！

## 4.3 基于信任区域的TRPO：加进KL散度解决两个分布相差大或步长难以确定的问题

好巧不巧，看似大功告成了，但重要性采样还是有个问题。具体什么问题呢，为更好的说明这个问题，我们回到上文的那个例子中。

> 还是那两个分布：$`p`$、$`q`$，当不能从$`p`$里面很好的采样数据，而能从$`q`$里面很好的采样数据时，基于重要性采样的原则，虽然我们可以把$`p`$换成任何的$`q`$，但是在实现上，$`p`$和$`q`$的差距不能太大，差距太大，会出问题
$`\mathbb{E}_{x \sim p}[f(x)]=\mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]`$
>
> 比如，虽然上述公式成立，但如果不是计算期望值，而是
>+  计算方差时$`Var_{x∼p}[f(x)]`$和$`Var_{x∼q}[f(x)\frac{p(x)}{q(x)}]`$是不一样的
> 因为两个随机变量的平均值相同，并不代表它们的方差相同
>
> 此话怎讲？以下是推导过程：
> 将$`f(x)`$、$`f(x) \frac{p(x)}{q(x)}`$分别代入方差的公式
```math
Var[X]=E[X^2]−(E[X])^2   
``` 
> 则分别可得(且考虑到不排除会有比初级更初级的初学者学习本文，故把第二个公式拆解的相对较细)   
```math
Var_{x \sim p}[f(x)]=\mathbb{E}_{x \sim p}\left[f(x)^{2}\right]-\left(\mathbb{E}_{x \sim p}[f(x)]\right)^{2}
```
```math
\begin{aligned} Var_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right] &=\mathbb{E}_{x \sim q}\left[\left(f(x) \frac{p(x)}{q(x)}\right)^{2}\right]-\left(\mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]\right)^{2} \\ &= \int \left(f(x) \frac{p(x)}{q(x)}\right)^{2} q(x) dx - \left(\mathbb{E}_{x \sim p}[f(x)]\right)^{2} \\ &= \int f(x)^{2} \frac{p(x)}{q(x)} p(x)dx - \left(\mathbb{E}_{x \sim p}[f(x)]\right)^{2} \\ &=\mathbb{E}_{x \sim p}\left[f(x)^{2} \frac{p(x)}{q(x)}\right]-\left(\mathbb{E}_{x \sim p}[f(x)]\right)^{2} \end{aligned}
```
>
> 上述两个公式前后对比，可以很明显的看出
>后者的第一项多乘了$`p(x)q(x)`$，如果$`p(x)q(x)`$差距很大，$`f(x)p(x)q(x)`$的方差就会很大  

所以结论就是，如果我们只要对分布$`p`$采样足够多次，对分布$`q`$采样足够多次，得到的期望值会是一样的。但是如果采样的次数不够多，会因为它们的方差差距可能是很大的，所以就可能得到差别非常大的结果。

这意味着什么呢，意味着我们目前得到的这个公式里
```math
J^{\theta^{\prime}}(\theta)=\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta^{\prime}}\left(a_{t} | s_{t}\right)} A^{\theta^{\prime}}\left(s_{t}, a_{t}\right)\right]
```

如果$`p_\theta(a_t|s_t)`$与$`p_{\theta}'(a_t|s_t)`$相差太多，即这两个分布相差太多，重要性采样的结果就会不好。怎么避免它们相差太多呢？这就是TRPO算法所要解决的。

2015年John Schulman等人提出了信任区域策略优化(Trust Region Policy Opimization，简称TRPO)，表面上，TRPO的出现解决了同时解决了两个问题，一个是解决重要性采样中两个分布差距太大的问题，一个是解决策略梯度算法中步长难以确定的问题

+   关于前者，在1.2.2节得到的目标函数基础上(下图第一个公式)，增加了一个KL散度约束(如下图第二个公式)

    $`J^{\theta^{\prime}}(\theta)=\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta^{\prime}}\left(a_{t} | s_{t}\right)} A^{\theta^{\prime}}\left(s_{t}, a_{t}\right)\right]`$

    $`\begin{aligned} J_{\mathrm{TRPO}}^{\theta^{\prime}}(\theta)=\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta^{\prime}}\left(a_{t} | s_{t}\right)} A^{\theta^{\prime}}\left(s_{t}, a_{t}\right)\right],\mathrm{KL}\left(\theta, \theta^{\prime}\right)<\delta \end{aligned}`$

    至此采样效率低效的问题通过重要性采样(重要性权重)、以及增加KL散度约束解决了

    > KL散度(KL divergence)，也称相对熵，而相对熵 = 交叉熵 - shannon熵，其衡量的是两个数据分布$`p`$和$`q`$之间的差异
    >
    > $`D_{KL}(P||Q) = E_x log\frac{P(x)}{Q(x)}`$
    >
    > 下图左半边是一组原始输入的概率分布曲线$`p(x)`$，与之并列的是重构值的概率分布曲线$`q(x)`$，下图右半边则显示了两条曲线之间的差异
    >
    > ![](./assets/images/RL_simple_primer/046b440033c8296f2f9b0f1bf5c3190e.png)
    >
    > 顺带从零推导下KL散度的公式
    >
    > 1 所谓**概率**：对于$`x`$，可以定义概率分布为$`p(x)`$或$`q(x)`$
    >
    > 2 所谓**信息**：对$`p(x)`$取对数，加符号得正值$`I(p) = -logp(x)`$，概率越高，包含的信息越小，因为事件越来越确定；相反，概率越低，包含的信息越多，因为事件具有很大的不确定性
    >
    > 3 所谓**Shannon熵**(熵是信息的平均，直观上，Shannon熵是信息在同一分布下的平均)：$`p(x)`$对$`I(p)`$平均，即
    >
    > $`\begin{aligned} H(p) &= E_{x\sim P} [I(p)] \\&= \sum p(x)I(p) \\&= - \sum p(x)logp(x) \end{aligned}`$
    >
    > 4 所谓**交叉熵Cross-Entropy**(直观上，交叉熵是信息在不同分布下的平均)，即指$`p(x)`$对$`I(q)`$平均，即
    >
    > $`\begin{aligned} H(p,q) &= E_{x\sim P} [I(q)] \\&= \sum p(x)I(q) \\&= - \sum p(x)logq(x) \end{aligned}`$
    >
    > 5 所谓相对熵或**KL散度 = 交叉熵 - shannon熵**，即
    >
    > $`\begin{aligned} D_{KL}(p||q) &= H(p,q) - H(p) \\&= -\sum p(x)logq(x) + \sum p(x)logp(x) \\&= -\sum p(x)log\frac{q(x)}{p(x)} \\&= \sum p(x)log\frac{p(x)}{q(x)} \end{aligned}`$
    >
    > 所以如果在KL散度表达式的最前面加个负号，再结合Jensen不等式自然有
    >
    > $`\begin{aligned} -D_{KL}(p||q) &= \sum p(x)log\frac{q(x)}{p(x)} \\& \leq log \sum p(x)\frac{q(x)}{p(x)} \\&= log1 \\&= 0 \end{aligned}`$

+   关于后者，具体而言，当策略网络是深度模型时，沿着策略梯度更新参数，很有可能由于步长太长，策略突然显著变差，进而影响训练效果

    这是1.2.1节，我们已经得到的策略梯度计算、策略梯度更新公式如下(别忘了，学习率$`\eta`$类似步长、距离的含义）分别如下

    $`\nabla \bar{R}_{\theta}=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)`$

    $`\theta \leftarrow \theta+\eta \nabla \bar{R}_{\theta}`$

    对这个问题，我们考虑在更新时找到一块信任区域(trust region)，在这个区域上更新策略时能够得到安全性保证，这就是TRPO算法的主要思想


本质上，其实这两个问题是同一个问题(简言之，**避免两个分布相差大即意味着避免步长过大**)。举个例子，比如爬两面都是悬崖的山，左右脚交替往前迈步，无论哪只脚向前迈步都是一次探索

1.  为尽快到达山顶且不掉下悬崖，一方面 你会选择最陡峭的方向，二方面 你会用目光选择一片信任区域，从而尽量远离悬崖边，在信任区域中，首先确定要探索的最大步长（下图的黄色圆圈），然后找到最佳点并从那里继续搜索
2.  好，现在问题的关键变成了，怎么确定每一步的步长呢？如果每一步的步长太小，则需要很长时间才能到达峰值，但如果步长太大，就会掉下悬崖(像不像两个分布之间差距不能太大)
3.  具体做法是，从初始猜测开始可选地，然后动态地调整区域大小。例如，如果新策略和当前策略的差异越来越大，可以缩小信赖区域。怎么实现？KL散度约束!

![](./assets/images/RL_simple_primer/d7fff6ebbf9a42d3ba9c04e1c924d476.png)

总之，TRPO就是考虑到连续动作空间无法每一个动作都搜索一遍，因此大部分情况下只能靠猜。如果要猜，就最好在信任域内部去猜。而TRPO将每一次对策略的更新都限制了信任域内，从而极大地增强了训练的稳定性。

至此，PG算法的采样效率低下、步长难以确定的问题都被我们通过TRPO给解决了。但TRPO的问题在哪呢？

TRPO的问题在于把 KL 散度约束当作一个额外的约束，没有放在目标里面，导致TRPO很难计算，总之因为信任域的计算量太大了，John Schulman等人于2017年又推出了TRPO的改进算法：PPO

## 4.4 近端策略优化PPO：解决TRPO的计算量大的问题

### 4.4.1 什么是近端策略优化PPO与PPO-penalty

如上所述，PPO算法是针对TRPO计算量的大的问题提出来的，正因为PPO基于TROP的基础上改进，故PPO也解决了策略梯度不好确定学习率Learningrate(或步长Stepsize)的问题。毕竟通过上文，我们已经得知

1.  如果stepsize过大,学出来的Policy会一直乱动，不会收敛；但如果StepSize太小，想完成训练，我们会等到地老天荒
2.  而PPO利用NewPolicy和OldPolicy的比例，限制了NewPolicy的更新幅度，让策略梯度对稍微大点的Stepsize不那么敏感

具体做法是，PPO算法有两个主要的变种：近端策略优化惩罚（PPO-penalty）和近端策略优化裁剪（PPO-clip），其中PPO-penalty和TRPO一样也用上了KL散度约束。

近端策略优化惩罚PPO-penalty的流程如下

1.  首先，明确目标函数，通过上节的内容，可知咱们需要优化$`J^{\theta'}(\theta)`$，让其最大化

    $`J^{\theta^{\prime}}(\theta)=\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_{\theta}\left(a_{t} \mid s_{t}\right)}{p_{\theta^{\prime}}\left(a_{t} \mid s_{t}\right)} A^{\theta^{\prime}}\left(s_{t}, a_{t}\right)\right]`$

2.  接下来，先初始化一个策略的参数$`\theta`$，在每一个迭代里面，我们用前一个训练的迭代得到的actor的参数$`\theta '`$与环境交互，采样到大量状态-动作对，根据$`\theta '`$交互的结果，估测$`A^{\theta '}(s_t,a_t)`$

3.  由于目标函数牵涉到重要性采样，而在做重要性采样的时候，$`p_\theta(a_t|s_t)`$不能与$`p_{\theta}'(a_t|s_t)`$相差太多，所以需要在训练的时候加个约束，这个约束就好像正则化的项一样，是$`\theta`$与$`\theta'`$输出动作的 KL散度，用于衡量$`\theta`$与$`theta'`$的相似程度，我们希望在训练的过程中，学习出的$`theta`$与$`theta'`$越相似越好
    所以需要最后使用 PPO 的优化公式：$`\\J_{\mathrm{PPO}}^{\theta^{\prime}}(\theta)=J^{\theta^{\prime}}(\theta)-\beta \mathrm{KL}\left(\theta, \theta^{\prime}\right)`$


    当然，也可以把上述那两个公式合二为一『如此可以更直观的看出，PPO-penalty把KL散度约束作为惩罚项放在了目标函数中(可用梯度上升的方法去最大化它)，此举相对TRPO减少了计算量』

    $`J_{\mathrm{PPO}}^{\theta^{\prime}}(\theta)=\mathbb{E}_{\left(s_{t}, a_{t}\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_{\theta}\left(a_{t} \mid s_{t}\right)}{p_{\theta^{\prime}}\left(a_{t} \mid s_{t}\right)} A^{\theta^{\prime}}\left(s_{t}, a_{t}\right)\right] \quad-\beta \mathrm{KL}\left(\theta, \theta^{\prime}\right)`$


上述流程有一个细节并没有讲到，即$`\beta`$是怎么取值的呢，事实上，$`\beta`$是可以动态调整的，故称之为自适应KL惩罚(adaptive KL penalty)，具体而言

* 先设一个可以接受的 KL 散度的最大值$`KL_{max}`$
假设优化完$`J_{\mathrm{PPO}}^{\theta^{\prime}}(\theta)=J^{\theta^{\prime}}(\theta)-\beta \mathrm{KL}\left(\theta, \theta^{\prime}\right)`$以后，**KL 散度值太大导致$`KL(\theta,\theta^{\prime})> KL_{max}`$，意味着$`\theta`$与$`\theta'`$差距过大(即学习率/步长过大)**，也就代表后面惩罚的项$`\beta \mathrm{KL}\left(\theta, \theta^{\prime}\right)`$惩罚效果太弱而没有发挥作用，**故增大惩罚把$`\beta`$增大**

* 再设一个 KL 散度的最小值$`KL_{min}`$
  如果优化完$`J_{\mathrm{PPO}}^{\theta^{\prime}}(\theta)=J^{\theta^{\prime}}(\theta)-\beta \mathrm{KL}\left(\theta, \theta^{\prime}\right)`$以后，
  KL散度值比最小值还要小导致$`KL(\theta ,\theta ^{\prime})< KL_{max}`$，意味着$`\theta`$与$`\theta'`$差距过小，也就代表后面惩罚的项$`\beta \mathrm{KL}\left(\theta, \theta^{\prime}\right)`$惩罚效果太强了，我们担心它只优化后一项，使$`\theta`$与$`\theta'`$一样，这不是我们想要的，所以减小惩罚即减小$`\beta`$

> 关于$`\beta`$具体怎么设置的？除了上面提到的自适应KL惩罚(adaptive KL penalty)，来自2017年发表的PPO论文
>
> ![](./assets/images/RL_simple_primer/2c3b63b5cb414b438df7a89e14dae8a4.png)
>
> 另外，2019年发表的论文《[Fine-Tuning Language Models from Human Preferences](https://arxiv.org/pdf/1909.08593 "Fine-Tuning Language Models from Human Preferences")》『*也是本博客内另一篇文章“[ChatGPT相关技术论文100篇](https://blog.csdn.net/v_JULY_v/article/details/129508065 "ChatGPT相关技术论文100篇")”中提到的第56篇，另这是论文对应的代码：[微调GPT2](https://github.com/openai/lm-human-preferences "微调GPT2")*』，其中也提到了根据 KL\_target 自适应调节$`\beta`$的算法，这个方法已经被 TRLX/TRL实现
>
> ![](./assets/images/RL_simple_primer/c3f775519533445db1120f7dc79d4ba1.png)
>
> 更多训练细节可以看下instructGPT论文原文
>
> ![](./assets/images/RL_simple_primer/dcf2240f8a56451089a314ffe0c6fc08.png)

总之，近端策略优化惩罚可表示为

$`\begin{aligned} &J_{\text{PPO}}^{\theta'}(\theta)=J^{\theta'}(\theta)-\beta \text{KL}\left(\theta, \theta'\right) \\ &J^{\theta'}(\theta) \approx \sum_{\left(s_{t}, a_{t}\right)} \frac{p_{\theta}\left(a_{t} \mid s_{t}\right)}{p_{\theta'}\left(a_{t} \mid s_{t}\right)} A^{\theta'}\left(s_{t}, a_{t}\right)\end{aligned}`$

### 4.4.2 PPO算法的另一个变种：近端策略优化裁剪PPO-clip

如果觉得计算 KL散度很复杂，则还有一个**PPO2**算法，即**近端策略优化裁剪PPO-clip**。近端策略优化裁剪的目标函数里面没有 KL 散度，其要最大化的目标函数为(easy RL上用$`\theta ^k`$代替$`\theta '`$，为上下文统一需要，本笔记的文字部分统一用$`\theta '`$)

$`\begin{aligned} J_{\mathrm{PPO2}}^{\theta'}(\theta) \approx \sum_{\left(s_{t}, a_{t}\right)} \min &\left(\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta'}\left(a_{t} | s_{t}\right)} A^{\theta'}\left(s_{t}, a_{t}\right),{clip}\left(\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta'}\left(a_{t} | s_{t}\right)}, 1-\varepsilon, 1+\varepsilon\right) A^{\theta'}\left(s_{t}, a_{t}\right)\right) \end{aligned}`$

整个目标函数在$`min`$这个大括号里有两部分，最终对比两部分那部分更小，就取哪部分的值，这么做的本质目标就是为了让$`p_{\theta }(a_{t}|s_{t})`$和$`p_{\theta'}(a_{t}|s_{t})`$可以尽可能接近，不致差距太大。

换言之，这个裁剪算法和KL散度约束所要做的事情本质上是一样的，都是为了让两个分布之间的差距不致过大，但裁剪算法相对好实现，别看看起来复杂，其实代码很好写

```python
// ratios即为重要性权重，exp代表求期望，括号里的environment_log_probs代表用于与环境交互的策略
ratios = torch.exp(log_probs - environment_log_probs)

// 分别用sur_1、sur_2来计算公式的两部分
// 第一部分是重要性权重乘以优势函数
sur_1 = ratios * advs

// 第二部分是具体的裁剪过程
sur_2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advs

// 最终看谁更小则取谁
clip_loss = -torch.min(sur_1,sur_2).mean()
```

回到公式，公式的第一部分我们已经见过了，好理解，咱们来重点分析公式的第二部分

$`{clip}\left(\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta'}\left(a_{t} | s_{t}\right)}, 1-\varepsilon, 1+\varepsilon\right) A^{\theta'}\left(s_{t}, a_{t}\right)`$

* 首先是$`{clip}`$括号里的部分，用一句话简要阐述下其核心含义就是：如果$`{p_{\theta}(a_{t}|s_{t})}`$和$`{p_{\theta'}(a_{t}|s_{t})}`$之间的概率比落在范围$`(1- \varepsilon)`$和$`(1+\varepsilon)`$之外，$`\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta'}\left(a_{t} | s_{t}\right)}`$将被剪裁，使得其值最小不小于$`(1- \varepsilon)`$，最大不大于$`(1+\varepsilon)`$

![](./assets/images/RL_simple_primer/1798baf5dba54e21a19508e82d407a8a.png)

* 然后是$`{clip}`$括号外乘以$`A^{\theta '}(s_t,a_t)`$，如果$`A^{\theta '}(s_t,a_t)`$大于0，则说明这是好动作，需要增大$`p_{\theta }(a_{t}|s_{t})`$，但$`\frac{p_{\theta}(a_{t}|s_{t})}{p_{\theta'}(a_{t}|s_{t})}`$最大不能超过$`(1+\varepsilon)`$  

如果$`A^{\theta '}(s_t,a_t)`$小于0，则说明该动作不是好动作，  
需要减小$`p_{\theta }(a_{t}|s_{t})`$，但$`\frac{p_{\theta }(a_{t}|s_{t})}{p_{\theta'}(a_{t}|s_{t})}`$最小不能小过$`(1-\varepsilon)`$

>  最后把公式的两个部分综合起来，针对整个目标函数

```math
\begin{aligned} J_{\mathrm{PPO2}}^{\theta'}(\theta) \approx \sum_{\left(s_{t}, a_{t}\right)} \min &\left(\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta'}\left(a_{t} | s_{t}\right)} A^{\theta'}\left(s_{t}, a_{t}\right),{clip}\left(\frac{p_{\theta}\left(a_{t} | s_{t}\right)}{p_{\theta'}\left(a_{t} | s_{t}\right)}, 1-\varepsilon, 1+\varepsilon\right) A^{\theta'}\left(s_{t}, a_{t}\right)\right) \end{aligned}
```

>  如果$`A^{\theta'}(s_t,a_t)`$大于0且$`\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)}`$大于$`(1+\epsilon)`$
>  则相当于第二部分是$`(1+\epsilon)×A^{\theta'}(s_t,a_t)`$，和第一部分$`\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)}*A^{\theta'}(s_t,a_t)`$对比取更小值当然是$`(1+\epsilon)`$的截断值： $`(1+\epsilon )*A^{\theta \prime}(s_t,a_t)`$

> 如果$`A^{\theta'}(s_t,a_t)'大于0且\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)}`$小于$`(1-\epsilon)`$
则相当于第二部分是$`(1-\epsilon)*A^{\theta'}(s_t,a_t)`$，和第一部分$`\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)}*A^{\theta'}(s_t,a_t)`$对比取更小值当然是原函数值： $`\frac{p_{\theta}(a_t|s_t)}{p_{\theta \prime}(a_t|s_t)}*A^{\theta \prime}(s_t,a_t)`$
> ![](./assets/images/RL_simple_primer/0bb3ab43b467ce1071d28a89537abc9c.png)
>
> 反之，**如果 $`A^{\theta \prime}(s_t,a_t)`$ 小于0，则最终目标函数的取值为了更小则和 $`A^{\theta \prime}(s_t,a_t)`$大于0时反过来**，毕竟加了个负号自然一切就不同了，为方便初学者一目了然，咱们还是把计算过程列出来，即
>+  如果$`A^{\theta'}(s_t,a_t)`$小于0且$`\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)}`$大于$`(1+\epsilon)`$
则相当于第二部分是$`(1+\epsilon)*A^{\theta'}(s_t,a_t)`$，和第一部分$`\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)}*A^{\theta'}(s_t,a_t)`$对比  
取更小值当然是原函数值： $`\frac{p_{\theta}(a_t|s_t)}{p_{\theta \prime}(a_t|s_t)}*A^{\theta \prime}(s_t,a_t)`$
>
>+   如果$`A^{\theta'}(s_t,a_t)`$小于0且$`\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)}`$小于$`(1-\epsilon)`$
则相当于第二部分是$`(1-\epsilon)×A^{\theta'}(s_t,a_t)`$，和第一部分$`\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)}*A^{\theta'}(s_t,a_t)`$对比取更小值当然是$`(1-\epsilon)`$的截断值： $`(1-\epsilon )*A^{\theta \prime}(s_t,a_t)`$

### 4.4.3 PPO算法的一个简单实现：对话机器人

综上，PPO算法是一种具体的Actor-Critic算法实现，比如在对话机器人中，输入的prompt是state，输出的response是action，想要得到的策略就是怎么从prompt生成action能够得到最大的reward，也就是拟合人类的偏好。具体实现时，可以按如下两大步骤实现

1.  **首先定义4个模型：Actor(action\_logits)、SFT(sft\_logits)、Critic(value)、RM「r(x, y)」，和kl\_div、reward、优势函数adv**
    从prompt库中采样出来的prompt在经过SFT(微调过GPT3/GPT3.5的模型称之为SFT)做generate得到一个response，这个『prompt + response』定义为sequence(这个采样的过程是批量采样进行generate，得到一个sequence buffer)，然后这个sequence buffer的内容做batched之后输入给4个模型做inference

    ![](./assets/images/RL_simple_primer/1ecb75833281415497f94e0cbe0279bd.png)

    这4个模型分别为Actor、SFT、Critic、RM，其中：
    Actor和SFT都是175B的模型，且Actor参数由SFT初始化(SFT是baseline)，Actor输出action\_logits，SFT输出sft\_logits
    sft\_logits和action\_logits做kl\_div，为了约束actor模型的更新step不要偏离原始模型SFT太远

    Critic和RM是6B的模型，Critic参数由RM初始化
    Critic输出标量value，RM输出标量r(x, y)，由r(x, y)和kl\_div计算得到reward，

    reward和value计算得到adv
2.  **其次，通过pg\_loss和value\_loss优化迭代**
    Actor的流程是取出sequence，然后inference生成新的logits，再和sequence对应的之前的logits计算ratio，和adv计算出pg\_loss，也就是actor的loss，然后反向传播，优化器迭代
    Critic的流程是取出sequence，然后inference得到新的value，和old\_value做**clip\_value，再和reward计算value loss**，然后反向传播，优化器迭代

    ![](./assets/images/RL_simple_primer/7d5efb19d49a44cbbdf7bf50da94712d.png)


至于代码实现可以参阅此文：[类ChatGPT逐行代码解读(2/2)：从零实现ChatLLaMA、ColossalChat、DeepSpeed Chat](https://blog.csdn.net/v_JULY_v/article/details/129996493 "类ChatGPT逐行代码解读(2/2)：从零实现ChatLLaMA、ColossalChat、DeepSpeed Chat")

## 后记

1.6日决定只是想写个ChatGPT通俗导论，但为了讲清楚其中的PPO算法，更考虑到之后还会再写一篇强化学习极简入门，故中途花了大半时间去从头彻底搞懂RL，最终把网上关于RL的大部分中英文资料都翻遍之外(详见参考文献与推荐阅读)，还专门买了这几本书以系统学习

+   第1本，《白话强化学习与pytorch》，偏通俗，对初学者友好，缺点是部分内容混乱，且对部分前沿/细节比如PPO算法阐述不足，毕竟19年出版的了
+   第2本，《EazyRL强化学习教程》，基于台大李宏毅等人的公开课，虽有不少小问题且其对MDP和三大表格求解法的阐述比较混乱，但其对策略梯度和PPO的阐述于初学入门而言还不错的
+   第3本，《动手学强化学习》，张伟楠等人著，概念讲解、公式定义都很清晰，且配套代码实现，当然 如果概念讲解能更细致则会对初学者更友好
+   第4本，《深度强化学习》，王树森等人著，偏原理讲解(无代码)，此书对于已对RL有一定了解的是不错的选择
+   第5本，《强化学习2》，权威度最高但通俗性不足，当然 也看人，没入门之前你会觉得此书晦涩，入门之后你会发现经典还是经典、不可替代，另书之外可配套七月的强化学习2带读课
+   第6本，《深度强化学习：基于Python的理论与实践》，理论讲的挺清楚，代码实践也不错
+   第7本，《强化学习(微课版)》，这本书是作为RL教材出版的，所以有教材的特征，即对一些关键定理会举例子展示实际计算过程，比如马尔可夫决策的计算过程，对初学者友好

总之，RL里面的概念和公式很多(相比ML/DL，RL想要机器/程序具备更好的自主决策能力)，而

+  一方面，绝大部分的资料没有站在初学者的角度去通俗易懂化、没有把概念形象具体化、没有把公式拆解举例化(如果逐一做到了这三点，何愁文章/书籍/课程不通俗)
+  二方面，不够通俗的话，则资料一多，每个人的公式表达习惯不一样便会导致各种形态，虽说它们最终本质上都是一样的，可当初学者还没有完全入门之前，就不一定能一眼看出背后的本质了，然后就不知道该以哪个为准，从而丧失继续前进的勇气(这种情况下，要么硬着头皮继续啃 可能会走弯路，要么通过比如七月在线的课程问老师或人 提高效率少走弯路)

   比如一个小小的策略梯度的计算公式会有近10来种表达，下面特意贴出其中6种，供读者辨别  

第一种，本笔记和Easy RL中用的
```math
\nabla \bar{R}_{\theta}=\frac{1}{N}\sum_{n=1}^N{\sum_{t=1}^{T_n}{R}}\left( \tau ^n \right) \nabla \log p_{\theta}\left( a_{t}^{n}|s_{t}^{n} \right) 
```
第二种，Sutton强化学习《Reinforcement Learning: An Introduction》第一版中用的

```math
\nabla _{\theta}J(\pi _{\theta})=\sum_s^{}{d^{\pi}}(s)\sum_a^{}{\nabla _{\theta}}\pi _{\theta}(a|s)Q^{\pi}(s,a)=E_{\pi}[\gamma ^t\nabla _{\theta}log\pi _{\theta}(A_t|S_t)Q^{\pi}(S_t,A_t)]
```

 其中
$`
d\pi (s)=\sum_{t=0}^{\infty}{\gamma ^t}Pr(s_0\rightarrow s,t,\pi )=\sum_{t=0}^{\infty}{\gamma ^t}Pr\left\{ S_t=s|s_0,\pi \right\} 
`$叫做discounted state distribution

第三种，David sliver在2014年的《Deterministic Policy Gradient Algorithm》论文中用的

```math
\nabla _{\theta}J(\pi _{\theta})=\int_S^{}{\rho ^{\pi}}(s)\int_A^{}{\nabla _{\theta}}\pi _{\theta}(a|s)Q^{\pi}(s,a)=E_{s\sim \rho ^{\pi},a\sim \pi _{\theta}}[\nabla _{\theta}log\pi _{\theta}(a|s)Q^{\pi}(s,a)]
```
其中，$`\rho ^{\pi}(s)`$与上述$`d\pi (s)`$相同，都是discounted state distribution。

第四种，肖志清《强化学习：原理与Python实现》中用的

   
```math
\nabla _{\theta}J(\pi _{\theta})=E[\sum_{t=0}^{\infty}{\gamma ^t}\nabla _{\theta}log\pi _{\theta}(A_t|S_t)Q^{\pi}(S_t,A_t)]
```
第五种，Sutton强化学习在2018年的第二版中用的

```math
\nabla _{\theta}J(\pi _{\theta})\propto \sum_S^{}{\mu ^{\pi}}(s)\sum_a^{}{\nabla _{\theta}}\pi _{\theta}(a|s)Q^{\pi}(s,a)=E_{\pi}[\nabla _{\theta}log\pi _{\theta}(A_t|S_t)Q^{\pi}(S_t,A_t)]

```

其中，
```math
\mu ^{\pi}(s)=\lim_{t\rightarrow \propto} Pr(S_t=s|s_0,\pi _{\theta})
```
是stationary distribution (undiscounted state distribution)

第六种，Open AI spinning up教程中用的

```math
\nabla _{\theta}J(\pi _{\theta})=E_{(\tau \sim \pi )}[\sum_{t=0}^T{\nabla _{\theta}}log\pi _{\theta}(a_t|s_t)Q^{\pi}(s_t,a_t)]

```

## 参考文献与推荐阅读

1.  关于强化学习入门的一些基本概念
2.  《白话强化学习与Pytorch》，此书让我1.6日起正式开启RL之旅，没看此书之前，很多RL资料都不好看懂
3.  《EazyRL强化学习教程》，基于台大李宏毅和UCLA周博磊等人的RL公开课所编著，其[GitHub](https://github.com/datawhalechina/easy-rl "GitHub")、[其在线阅读地址](https://datawhalechina.github.io/easy-rl/#/ "其在线阅读地址")
4.  《动手学强化学习》，张伟楠等人著
5.  台大李宏毅RL公开课，这是其：[视频/PPT/PDF下载地址](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html "视频/PPT/PDF下载地址")
6.  UCLA周博磊RL公开课，这是其[：GitHub地址](https://github.com/zhoubolei/introRL "：GitHub地址")
7.  关于Q-leaning的几篇文章：1[如何用简单例子讲解 Q - learning 的具体过程？](https://www.zhihu.com/question/26408259 "如何用简单例子讲解 Q - learning 的具体过程？")2[莫烦：什么是 Q-Learning](https://zhuanlan.zhihu.com/p/24808797 "莫烦：什么是 Q-Learning")
8.  AlphaGo作者之一David Silver主讲的[增强学习笔记1](https://zhuanlan.zhihu.com/p/50478310 "增强学习笔记1")、[笔记2](https://www.zhihu.com/column/reinforce "笔记2")，另附其讲授的《UCL Course on RL》的[课件地址](https://www.davidsilver.uk/teaching/ "课件地址")
9.  huggingface的两篇RL教程：[An Introduction to Deep Reinforcement Learning](https://huggingface.co/blog/deep-rl-intro "An Introduction to Deep Reinforcement Learning")、[GitHub：The Hugging Face Deep Reinforcement Learning Course](https://github.com/huggingface/deep-rl-class "GitHub：The Hugging Face Deep Reinforcement Learning Course")

10.  TRPO原始论文：[Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf "Trust Region Policy Optimization")
11.  **PPO原始论文**：[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf "Proximal Policy Optimization Algorithms")
12.  PPO算法解读(英文2篇)：解读1 [RL — Proximal Policy Optimization (PPO) Explained](https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12 "RL — Proximal Policy Optimization (PPO) Explained")、解读2[Proximal Policy Optimization (PPO)](https://huggingface.co/blog/deep-rl-ppo "Proximal Policy Optimization (PPO)")
13.  PPO算法解读(中文4篇)：[Easy RL上关于PPO的详解](https://datawhalechina.github.io/easy-rl/#/chapter5/chapter5 "Easy RL上关于PPO的详解")、[详解近端策略优化](https://www.cnblogs.com/xingzheai/p/15931681.html "详解近端策略优化")、[详解深度强化学习 PPO算法](https://zhuanlan.zhihu.com/p/88525394?utm_id=0 "详解深度强化学习 PPO算法")、[ChatGPT第二弹：PPO算法](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650435573&idx=2&sn=427f3ad2cb6ab5120686652fd9a64b8f&chksm=becde9af89ba60b9a6a4306b126b32deff8a78788208fd431a40f89736390c6883f35dc0b71d&mpshare=1&scene=23&srcid=0208gSWRajh40YzsiJ2YqN2B&sharer_sharetime=1675828126672&sharer_shareid=8dff0e13d79dbe85e759d04101e63bbf#rd "ChatGPT第二弹：PPO算法")
14.  PPO算法实现：[GitHub - lvwerra/trl: Train transformer language models with reinforcement learning.](https://github.com/lvwerra/trl "GitHub - lvwerra/trl: Train transformer language models with reinforcement learning.")
15.  [如何选择深度强化学习算法？MuZero/SAC/PPO/TD3/DDPG/DQN/等](http://www.deeprlhub.com/d/166-muzerosacppotd3ddpgdqn "如何选择深度强化学习算法？MuZero/SAC/PPO/TD3/DDPG/DQN/等")
16.  [如何通俗理解隐马尔可夫模型HMM？](https://www.julyedu.com/questions/interview-detail?kp_id=30&cate=NLP&quesId=2765 "如何通俗理解隐马尔可夫模型HMM？")
17.  [HMM学习最佳范例](https://www.52nlp.cn/hmm%E5%AD%A6%E4%B9%A0%E6%9C%80%E4%BD%B3%E8%8C%83%E4%BE%8B%E5%85%A8%E6%96%87pdf%E6%96%87%E6%A1%A3%E5%8F%8A%E7%9B%B8%E5%85%B3%E6%96%87%E7%AB%A0%E7%B4%A2%E5%BC%95 "HMM学习最佳范例")
18.  [强化学习中“策略梯度定理”的规范表达、推导与讨论](https://zhuanlan.zhihu.com/p/490373525?utm_campaign=&utm_medium=social&utm_oi=644502718257958912&utm_psn=1602107506024824832&utm_source=qq "强化学习中“策略梯度定理”的规范表达、推导与讨论")
19.  [强化学习——时序差分算法](https://zhuanlan.zhihu.com/p/34747205?utm_campaign=&utm_medium=social&utm_oi=644502718257958912&utm_psn=1602473260729393152&utm_source=qq "强化学习——时序差分算法")
20.  [KL-Divergence详解](https://zhuanlan.zhihu.com/p/425693597?utm_id=0 "KL-Divergence详解")
21.  《强化学习(微课版)》，清华大学出版社出版
22.  [使用蒙特卡洛计算定积分（附Python代码）](https://www.guanjihuan.com/archives/1145 "使用蒙特卡洛计算定积分（附Python代码）")
23.  [David Silver 增强学习补充——重要性采样](https://zhuanlan.zhihu.com/p/78720910 "David Silver 增强学习补充——重要性采样")、[强化学习中的重要性采样(Importance Sampling)](https://zhuanlan.zhihu.com/p/371156865 "强化学习中的重要性采样(Importance Sampling)")
24.  [关于Instruct GPT复现的一些细节与想法](https://zhuanlan.zhihu.com/p/609078527 "关于Instruct GPT复现的一些细节与想法")
25.  [类ChatGPT逐行代码解读(2/2)：从零起步实现ChatLLaMA和ColossalChat](https://blog.csdn.net/v_JULY_v/article/details/129996493 "类ChatGPT逐行代码解读(2/2)：从零起步实现ChatLLaMA和ColossalChat")

* * *

## 附录：修改/完善/新增记录

RL里的细节、概念、公式繁多，想完全阐述清楚是不容易的，以下是自从23年1.16日以来的修改、完善、新增记录：

1.  1.16日，第一轮修改/完善/新增
    修正几个笔误，且考虑到不排除会有比初级更初级的初学者阅读本文，补充部分公式的拆解细节
2.  1.17日，先后修正了2.2节重要性采样与重要性权重的部分不准确的描述、修正个别公式的笔误，以及补充1.4.2中关于PPO-clip的细节阐述、优化第四部分的相关描述
3.  1.18日，为措辞更准确，优化1.2.5节“基于信任区域的TRPO：通过KL散度解决两个分布相差大和步长难以确定的问题”的相关描述
4.  1.19日，为让读者理解起来更一目了然
    优化1.3.1节中关于什么是近端策略优化PPO的描述
    优化1.3.2节中关于“近端策略优化惩罚PPO-penalty关于自适应KL惩罚（adaptive KL penalty）”的描述
    拆解细化关于
    $`\nabla \log p_{\theta}(\tau )`$的推导过程
    补充说明对优势函数
    $`A^{\theta}(s_t,a_t)`$的介绍
5.  1.20日，第五轮修改/完善/新增
    通过LaTeX重新编辑部分公式，以及补充说明1.2.1节中关于某一条轨迹$`\tau `$发生概率的定义
6.  1.21日(大年三十)，新增对蒙/新增特卡洛方法的介绍，以及新增$`R(\tau )-b`$中基线$`b`$的定义，且简化2.1.1节中关于强化学习过程的描述
7.  1.22日，为严谨起见改进第二部分中对回报$`G`$、状态价值函数、动作价值函数、马尔可夫决策、策略评价函数的描述，并纠正个别公式的笔误
8.  1.23日，梳理1.1节的整体内容结构和顺序脉络，以让逻辑更清晰，补充了MDP的整个来龙去脉(包括且不限于马尔可夫过程、马尔可夫奖励、马尔可夫决策以及贝尔曼方程)
9.  1.25日，为方便对高数知识有所遗忘的同学更顺畅的读懂本文，新增对导数、期望的简单介绍(后汇总至概率统计极简入门笔记里)，以及补充对$`R(\tau )-b`$中基线$`b`$的定义的介绍
10.  1.26日，第十轮修改/完善/新增
    优化改进2.2节关于策略梯度的梯度计算的整个推导过程，以让逻辑更清晰
11.  1.27日，优化关于增加基线baseline和优势函数等内容的描述
    在后记里补充策略梯度计算公式的5种其它表达
    优化关于“近端策略优化惩罚PPO-penalty的流程”的描述
12.  1.28日，新增对MC和TD方法各自的阐述及两者的对比，优化对KL散度定义的描述，新增近端策略优化裁剪PPO-clip的关键代码
13.  1.30日，新增马尔可夫决策的贝尔曼方程以及对应的计算图解，以方便一目了然
    简单阐述了下GPT2相比GPT的结构变化，以及完善丰富了下文末的参考文献与推荐阅读，比如增加图解GPT2、图解GPT3的参考文献
14.  1.31日，为行文严谨，针对1.1.2节中关于马尔可夫奖 励的部分  
    规范统一个别公式的大小写表示  
    补充状态$`s`$下奖励函数的定义$`R(s)=E[R_{t+1}|S_t=s]`$  
    修正回报公式的笔误$`G_t=R_{t+1}+\gamma \cdot R_{t+2}+\gamma ^2\cdot R_{t+3}+\gamma ^3\cdot R_{t+4}+\cdots `$  
    修正状态价值函数公式的笔误  
    且为形象起见，新增一个“吃饭-抽烟/剔牙”的例子以展示利用贝尔曼方程计算的过程    
    此外，且为通俗细致，针对1.1.3节中关于马尔科夫决策的部分
    拆解状态价值函数、动作价值函数的定义公式，拆解关于状态价值函数和动作价值函数之间联系的推导过程

15.  2.1日，第十五轮修改/完善/新增
    参考sutton RL一书，补充对奖励函数、回报公式、轨迹定义的公式表达规范的说明
16.  2.12日，为更一目了然，新增对状态价值函数的贝尔曼方程的解释，与例子说明
17.  2.13日，开始完善动态规划一节的内容，然后发现之前写的一篇关于DP的博客不够通俗，故本周得先修订下那篇旧文，以通俗理解DP
18.  2.15日，基于已经修订好的DP旧文，完善本文对DP算法的阐述
    并修正第一部分里关于“什么是强化学习”的不够准确的表述
19.  2.16日，新增对蒙特卡洛方法和时序差分法的介绍，并重点对比DP、MC、TD三者的区别与联系
    明确全文结构，分为四大部分，一一对应：RL基础、RL进阶、RL深入、RL高级
20.  2.17日，第二十轮修改/完善/新增
    新增第三部分价值学习：从n步Sarsa算法到Q-learning、DQN，并修订部分细节
    润色部分细节，修订部分目录
21.  2.20日，新增一节“RL的分类：基于模型(Value-base/Policy-based)与不基于模型”
22.  2.21日，新增KL散度公式的从零推导过程
23.  2.24日，新增关于$`E[G_{t+1}|S_t = s]`$为何等于$`E[V(S_{t+1}|S_t = s)]`$的详细推导
24.  2.26日，新增关于$`\nabla f(x)=f(x)\nabla \log f(x)`$如何而来的推导
25.  3.4日，完善对于“Q学习的两种策略：行为策略和目标策略”的描述
26.  2年4月，纠正本文下面几个热心读者指出的笔误，大家有啥问题 继续随时反馈，非常感谢
27.  4.27，针对一读者留言所表达的疑惑，新增1.2.2节中关于「奖励函数$`R(s,a)`$推导」的说明解释
    并新增一节“4.4.3 PPO算法的一个简单实现：对话机器人”
