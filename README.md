<!--
 * @Author: zh2nlp 17888822698@163.com
 * @Date: 2023-06-05 15:09:17
 * @LastEditors: zh2nlp 17888822698@163.com
 * @LastEditTime: 2023-06-05 15:19:23
 * @FilePath: \ChatGPT_principle_fine-tuning_code_paper\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

## Project background
ChatGPT一经推出便火爆全球，为了彻底写清楚ChatGPT背后的所有关键细节，July从1月初写到5月底仍未完工，过程中涉及到多篇文章(RL 论文 项目 CV多模态)，再加上之前写的Transformer、RL数学基础等多篇笔记，成了一个大系列：

- [ChatGPT技术原理解析：从RL之PPO算法、RLHF到GPT4、instructGPT](https://github.com/julycoding/ChatGPT_principle_fine-tuning_code_paper/blob/main/ChatGPT%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86%E8%A7%A3%E6%9E%90%EF%BC%9A%E4%BB%8ERL%E4%B9%8BPPO%E7%AE%97%E6%B3%95%E3%80%81RLHF%E5%88%B0GPT4%E3%80%81instructGPT.md)
- [Transformer通俗笔记：从Word2Vec、Seq2Seq逐步理解到GPT、BERT](https://github.com/julycoding/ChatGPT_principle_fine-tuning_code_paper/blob/main/Transformer%E9%80%9A%E4%BF%97%E7%AC%94%E8%AE%B0%EF%BC%9A%E4%BB%8EWord2Vec%E3%80%81Seq2Seq%E9%80%90%E6%AD%A5%E7%90%86%E8%A7%A3%E5%88%B0GPT%E3%80%81BERT.md)
- RL所需的微积分/概率统计基础、最优化基础
- [强化学习极简入门(上)：通俗理解MDP、DP MC TC和Q学习](https://github.com/julycoding/ChatGPT_principle_fine-tuning_code_paper/blob/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E6%9E%81%E7%AE%80%E5%85%A5%E9%97%A8%EF%BC%9A%E9%80%9A%E4%BF%97%E7%90%86%E8%A7%A3MDP%E3%80%81DP%20MC%20TC%E5%92%8CQ%E5%AD%A6%E4%B9%A0%E3%80%81%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E3%80%81PPO.md)
- [强化学习极简入门(下)：策略梯度、PPO](https://github.com/julycoding/ChatGPT_principle_fine-tuning_code_paper/blob/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E6%9E%81%E7%AE%80%E5%85%A5%E9%97%A8%E4%B8%8B%EF%BC%9A%E9%80%9A%E4%BF%97%E7%90%86%E8%A7%A3MDP%E3%80%81DP%20MC%20TC%E5%92%8CQ%E5%AD%A6%E4%B9%A0%E3%80%81%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E3%80%81PPO.md) 
- ChatGPT与多模态必读论文100篇(2.27日起，每周更新)
- 类ChatGPT的部署与微调：从LLaMA、Alpaca/Vicuna/BELLE、中文版、从GLM、ChatGLM到MOSS、ChatDoctor、可商用
- 类ChatGPT代码逐行解读：从零实现Transformer、ChatGLM-6B、从零实现TRL、ChatLLaMA、ColossalChat、DeepSpeed Chat
- AI绘画与CV多模态原理解析：VAE、扩散模型DDPM、DETR、ViT/Swin transformer、CLIP/BLIP到stable diffusion、GPT4(后者待6月中旬发布)

————————————————

23年5月9日，七月ChatGPT原理解析课的一学员虞同学在群内建议道：“或者我们自己是否也可以搞一个项目，大家共同参与科研维护”，之后多位同学响应表示支持

July个人觉得也可以：“比如项目之一，可以先以我博客内部分文章 搬到GitHub上，然后维护修改旧的章节、扩写新的章节，再之后共同开发论文修订助手之类的项目(针对未成形的论文 通过微调某个开源模型 给出修订、审阅意见)”，于此便有了本GitHub：ChatGPT资源库(原理/微调/代码/论文)

## 100 papers
LLM/ChatGPT与多模态必读论文150篇(已更至第101篇)

https://blog.csdn.net/v_JULY_v/article/details/129508065

## co-sponsor

July、七月ChatGPT原理课的十几位同学，他们是：@corleytd、@EdwardSelf、@JusticeGL、@wangzaistone、@windspin2003、@zh2nlp..

----

## 本项目编写规范(初步)
温馨提示：由于本项目中存在大量的LaTex公式, github 与 原生markdown 适配可能有所差别，故如果将本项目clone到本地阅读可能导致在如Typro 等编辑软件中会出现显示异常，我们建议在github网页中进行浏览。

如果您在参与本项目内容贡献过程中遇到问题，可参考以下栏目，也可将新的解决方案或建议列在以下栏目中。

### 关于LaTex公式：
由于Github对于Markdown 原生语法中LaTex公式解析存在的部分缺憾，导致使用Markdown语法书写的数学公式在github网页中展示会出现异常，特于此文档当前栏目记录一些常用的手法，仅供参考。
[Github LaTex 支持文档](https://docs.github.com/zh/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
* 关于行内公式的书写手法：
  * 原生markdown采用"\$LaTex\$"包裹的形式
  * Github 中采用"\$\`LaTex\`\$"的形式进行包裹[此方案仅为解决网页版不显示的问题，在这种方案下，github 公式显示正常，但是原生的markdown中会出现多余字符，如有更好的方案，可直接在此处更新方案]
* 关于行间公式：
  * 原生markdown采用"\$\$LaTex\$\$"包裹的形式
  * Github 中采用以下形式包裹：
 <div>
 <p>
 ```math<br/>
  Latex<br/>
 ```<br/>
 </p>
</div>

* 关于常用的特殊字符公式：

|  符号     |   LaTeX   |    备注   | 
| :-----:   | :-------: |  :-------: | 
|   $`\#`$    |   `\#`    |             |
|   $`\%`$    |   `\%`    |             |
| $`^\wedge`$ | `^\wedge` |             |
|   $`\&`$    |   `\&`    |             |
|   $`\_`$    |   `\_`    |             |
|   $`\prime`$    |   `\prime`    |  导数           |
|   $`\lt`$    |   `\lt`    |  小于           |
|   $`\le`$    |   `\le`    |  小于等于           |
|   $`\gt`$    |   `\gt`    |  大于           |
|   $`\ge`$    |   `\ge`    |  大于等于           |
|   $`\mid`$    |   `\mid`    |  条件概率中的竖分割线           |
|   $`\cdots`$    |   `\cdots`    |  垂直居中省略号           |
|   $`\ldots`$    |   `\ldots`    |  底部对齐省略号           |
|   $`\omega`$    |   `\omega`    |             |
|   $`\Omega`$    |   `\Omega`    |             |
|   $`\lim \limits_{x \to \infty} f(x)`$    |   `\lim \limits_{x \to \infty} f(x)`    |      极限下标       |
|   $`PPO _{\theta} `$    |   `PPO_{\theta}`    |      普通下标       |
|   $`\sum \limits_{i = 1} ^ N `$    |   `\sum \limits_{i = 1} ^ N`    |      求和中的上下标       |
。。。


**备注**：Github中对于某些复杂的LaTex语法暂未支持，如果遇到渲染不出来的情况请酌情修改公式写法。


### 关于图片
本项目中的所有图片均保存在assets/images/doc_name 目录下。

---
