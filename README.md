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

July个人觉得也可以：“比如项目之一，可以先以我博客内部分文章 搬到GitHub上，然后维护修改旧的章节、扩写新的章节，再之后共同开发LLM对话机器人之类的项目”，于此便有了本GitHub：ChatGPT资源库(原理/微调/代码/论文)

## 100 papers
第一部分 OpenAI/Google的基础语言大模型(11篇，总11篇)
- Improving Language Understanding by Generative Pre-TrainingGPT原始论文
- Language Models are Unsupervised Multitask LearnersGPT2原始论文  
- Language Models are Few-Shot LearnersGPT3原始论文  
- Training language models to follow instructions with human feedbackInstructGPT原始论文
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer，19年10月，Google发布T5模型(transfer text to text transformer)，虽也基于transformer，但区别于BERT的编码器架构与GPT的解码器架构，T5是transformer的encoder-decoder架构
- LaMDA: Language Models for Dialog Applications论文发布于22年1月，显示LaMDA的参数高达137B，用的transformer decoder架构21年5月，Google对外宣布内部正在研发对话模型LaMDA，基于transformer decoder架构
Finetuned Language Models Are Zero-Shot Learners21年9月，Google提出FLAN大模型，其基于LaMDA-PT做Instruction Fine-Tuning
- PaLM: Scaling Language Modeling with Pathways22年3月，Google的Barham等人发布了Pathways系统，用于更高效地训练大型模型
- Constitutional AI: Harmlessness from AI FeedbackOpenAI之前一副总裁离职推出了ChatGPT的竞品最强竞品：Claude，ChatGPT用人类偏好训练RM再RL(即RLHF)，Claude则基于AI偏好模型训练RM再RL(即RLAIF)
- Improving alignment of dialogue agents via targeted human judgementsDeepMind的Sparrow，这个工作发表时间稍晚于instructGPT，其大致的技术思路和框架与 instructGPT 的三阶段基本类似，但Sparrow 中把奖励模型分为两个不同 RM 的思路  
- GPT-4 Technical Report增加了多模态能力的GPT4的技术报告  

第二部分 LLM的关键技术：ICL/CoT/RLHF/微调/词嵌入/位置编码/加速/与KG结合等(34篇，总45篇)
- Attention Is All You NeedTransformer原始论文  
- Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
- Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers这篇文章则将ICL看作是一种隐式的Fine-tuning
- A Survey on In-context Learning
- Noisy Channel Language Model Prompting for Few-Shot Text Classification
- MetaICL: Learning to Learn In Context
https://github.com/dqxiu/ICL_PaperList in-context learning研究梳理In-Context Learning到底有没有Learning？
- Meta-learning via Language Model In-context Tuning
- Evaluating Large Language Models Trained on CodeCodex原始论文
- Chain-of-Thought Prompting Elicits Reasoning in Large LanguageCoT原始论文，也从侧面印证，instructGPT从22年1月份之前 就开始迭代了
- Large Language Models are Zero-Shot Reasoners来自东京大学和谷歌的工作，关于预训练大型语言模型的推理能力的探究，“Let's think step by step”的梗即来源于此篇论文
- Emergent Abilities of Large Language ModelsGoogle 22年8月份发的，探讨大语言模型的涌现能力
- Multimodal Chain-of-Thought Reasoning in Language Models23年2月，亚马逊的研究者则在这篇论文里提出了基于多模态思维链技术改进语言模型复杂推理能力的思想
- TRPO论文
- Proximal Policy Optimization Algorithms2017年，OpenAI发布的PPO原始论文
- RLHF原始论文
- Scaling Instruction-Finetuned Language Models微调PaLM-540B(2022年10月)从三个方面改变指令微调，一是改变模型参数，提升到了540B，二是增加到了1836个微调任务，三是加上Chain of thought微调的数据
- The Flan Collection: Designing Data and Methods for Effective Instruction Tuning
- Fine-Tuning Language Models from Human Preferences
- LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELSLoRA论文
- P-Tuning微调论文
- Distributed Representations of Sentences and DocumentsMikolov首次提出 Word2vecEfficient estimation of word representations in vector spaceMikolov专门讲训练 Word2vec 中的两个trick：hierarchical softmax 和 negative sampling
- word2vec Explained- Deriving Mikolov et al.’s Negative-SamplingWord-Embedding MethodYoav Goldberg关于word2vec的论文，对 negative-sampling 的公式推导非常完备word2vec Parameter Learning ExplainedXin Rong关于word2vec的论文，非常不错
- ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING旋转位置嵌入(RoPE)论文
- ​​​​​​​Linearized Relative Positional Encoding统一了适用于linear transformer的相对位置编码
- SEARCHING FOR ACTIVATION FUNCTIONSSwiGLU的原始论文
《The Natural Language Decathlon:Multitask Learning as Question Answering》GPT-1、GPT-2论文的引用文献，Salesforce发表的一篇文章，写出了多任务单模型的根本思想
- Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916, 2022
- ZeRO: Memory Optimizations Toward Training Trillion Parameter ModelsZeRO是微软deepspeed的核心
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LMMegatron-LM 论文原始论文
Efficient sequence modeling综述包含sparse transformer、linear transformer(cosformer，transnormer）RNN(RWKV、S4)，Long Conv(TNN、H3）
- Vicuna tackle the memory pressure by utilizing gradient checkpointing and flash attentionTraining Deep Nets with Sublinear Memory Cost
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- Unifying Large Language Models and Knowledge Graphs: A RoadmapLLM与知识图谱的结合实战

第三部分 Meta等公司发布的类ChatGPT开源模型和各种微调(7篇，总52篇)
- LLaMA: Open and Efficient Foundation Language Models2023年2月24日Meta发布了全新的65B参数大语言模型LLaMA，开源，大部分任务的效果好于2020年的GPT-3
- SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions23年3月中旬，斯坦福发布Alpaca：只花100美元，人人都可微调Meta家70亿参数的LLaMA大模型，而斯坦福团队微调LLaMA的方法，便是来自华盛顿大学Yizhong Wang等去年底提出的这个Self-Instruct
- Alpaca: A Strong Open-Source Instruction-Following Model
- Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022
- BLOOM: A 176B-Parameter Open-Access Multilingual Language Model
- GLM: General Language Model Pretraining with Autoregressive Blank Infilling2022年5月，正式提出了GLM框架
- GLM-130B: AN OPEN BILINGUAL PRE-TRAINED MODELGLM-130B便是基于的GLM框架的大语言模型

第四部分 具备多模态能力的大语言模型(11篇，总63篇)
- BEiT: BERT Pre-Training of Image Transformers
- BEiT-2: Masked Image Modeling with Vector-Quantized Visual Tokenizers
- Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks2022年8月，微软提出的多模态预训练模型BEiT-3
- Language Is Not All You Need: Aligning Perception with Language Models微软23年3月1日发布的多模态大语言模型Kosmos-1的论文
- PaLM-E: An Embodied Multimodal Language Model(论文地址)Google于23年3月6日发布的关于多模态LLM：PaLM-E，可让能听懂人类指令且具备视觉能力的机器人干活
- Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models微软于23年3月8日推出visual ChatGPT
- MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
- ​Flamingo: a visual language model for few-shot learning
- Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022
- Language models are unsupervised multitask learners. 2019
- Improving language understanding by generative pre-training. 2018

第五部分 AI绘画与多模态能力背后的核心技术(21篇，总84篇)
- End-to-End Object Detection with TransformersDETR by 2020年5月
- AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION A​​​​​​T SCALE发表于2020年10月的Vision Transformer原始论文，代表Transformer正式杀入CV界
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows发表于21年3月
- Swin Transformer V2: Scaling Up Capacity and Resolution
- Auto-Encoding Variational Bayes
- Denoising Diffusion Probabilistic Models2020年6月提出DDPM，即众人口中常说的diffusion model
- Diffusion Models Beat GANs on Image Synthesis使用classifier guidance的方法，引导模型进行采样和生成
- High-Resolution Image Synthesis with Latent Diffusion Models2022年8月发布的Stable Diffusion基于Latent Diffusion Models，专门用于文图生成任务
- Aligning Text-to-Image Models using Human FeedbackChatGPT的主要成功要归结于采用RLHF来精调LLM，近日谷歌AI团队将类似的思路用于文生图大模型：基于人类反馈（Human Feedback）来精调Stable Diffusion模型来提升生成效果
- CLIP: Connecting Text and Images - OpenAI这是针对CLIP论文的解读之一  CLIP由OpenAI在2021年1月发布，超大规模模型预训练提取视觉特征，图片和文本之间的对比学习
- Zero-Shot Text-to-Image GenerationDALL·E原始论文
- Hierarchical Text-Conditional Image Generation with CLIP LatentsDALL·E 2论文2022年4月发布(至于第一代发布于2021年初)，通过CLIP + Diffusion models，达到文本生成图像新高度
- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models by Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi.  
- InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning23年5月发布的InstructBLIP论文
- LAVIS: A Library for Language-Vision IntelligenceSalesforce开源一站式视觉语言学习框架LAVIS，这是其GitHub地址：https://github.com/salesforce/LAVIS
- MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models对各种多模态模型的评测
- Segment Anything23年4.6日，Meta发布史上首个图像分割基础模型SAM，将NLP领域的prompt范式引进CV，让模型可以通过prompt一键抠图。网友直呼：CV不存在了!  
- A Comprehensive Survey on Segment Anything Model for Vision and Beyond对分割一切模型SAM的首篇全面综述：28页、200+篇参考文献
- Fast Segment Anything中科院版的分割一切
- MobileSAM比SAM小60倍，比FastSAM快4倍，速度和效果双赢
  
第六部分 预训练模型的发展演变史(3篇，总87篇)
- A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT预训练基础模型的演变史
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing

第七部分 垂域版类ChatGPT(比如医疗GPT)和其它(11篇，总98篇)
- Large Language Models Encode Clinical Knowledge​
- Towards Expert-Level Medical Question Answering with Large Language Models继上篇论文提出medpalm之后，5月16日，Google Research和DeepMind发布了Med-PaLM 2，相比第一代最显著的改进是基座模型换成了Google的最新大模型PaLM2(据说有着340b参数，用于训练的token数达3.6万亿)
- ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge医疗ChatDoctor论文
- BloombergGPT: A Large Language Model for Finance金融BloombergGPT论文
- Deep Residual Learning for Image RecognitionResNet论文，短短9页，Google学术被引现15万多
- WHAT LEARNING ALGORITHM IS IN-CONTEXT LEARNING? INVESTIGATIONS WITH LINEAR MODELS
- Transformer-XL: Attentive language models beyond a fixed-length context
- An empirical analysis of compute-optimal large language model training
- Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond
- COLT5: Faster Long-Range Transformers with Conditional Computation
- Offsite-Tuning: Transfer Learning without Full Model

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
