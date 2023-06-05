## 项目编写规范（初步）

### [关于本项目](README.md)

---
### 关于Latex公式：
由于Github对于Markdown 原生语法中Latex公式解析存在的部分缺憾，导致使用Markdown语法书写的数学公式在github网页中展示会出现异常，特于此文档当前栏目记录一些常用的手法，仅供参考。
[Github Latex 支持文档](https://docs.github.com/zh/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
* 关于行内公式的书写手法：
  * 原生markdown采用"\$Latex\$"包裹的形式
  * Github 中采用"\$\`Latex\`\$"的形式进行包裹[此方案仅为解决网页版不显示的问题，在这种方案下，github 公式显示正常，但是原生的markdown中会出现多余字符，如有更好的方案，可直接在此处更新方案]
* 关于行间公式：
  * 原生markdown采用"\$\$Latex\$\$"包裹的形式
  * Github 中采用以下形式包裹：
```math
    ```math
    Latex
    ```
```
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


**备注**：Github中对于某些复杂的Latex语法暂未支持，如果遇到渲染不出来的情况请酌情修改公式写法。

---

### 关于图片
本项目中的所有图片均保存在assets/images/doc_name 目录下。