---
layout: center
---
# Дополнительные слайды

---
zoom: 0.84
---

# Батч
<div></div>

**Батч**-подход тесно связан с градиентными методами. Батчи (пакеты) позволяют параллельное обучение DL сетей (то есть **значительно** сократить время обучения).

Общие рекомендации по размеру **минибатчей** [[*Deep Learning* by Ian Goodfellow et al., Chapter 8.1.3](https://www.deeplearningbook.org/)]:
* Большие партии дают более точную оценку градиента, но при этом
меньше, чем линейная отдача.
* Некоторые виды аппаратного обеспечения обеспечивают лучшее время работы с массивами определенных размеров
* Малые партии могут обеспечить регуляризирующий эффект.

Также очень важно, чтобы минибатчи были выбраны случайным образом.

Реализовать партии в PyTorch можно с помощью [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

**Нормализация батча** - это шаг гиперпараметра $\gamma, \beta$, который нормализует партию $\{x_i\}$. Отмечая $\mu_B, \sigma_B^2$ среднее и дисперсию того, что мы хотим скорректировать в партии, это делается следующим образом:

$$x_i \leftarrow \gamma ~\frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

---

# Скорость обучения

<div>
    <img src="/Learning_Rate.png" style="width: 500px; position: relative">
</div>
<br>
<div>
  <figure>
    <img src="/Learning_Rate_Loss.png" style="width: 400px; position: relative">
    <figcaption style="color:#b3b3b3ff; font-size: 11px">Источник изображения:
      <a href="https://www.jeremyjordan.me/nn-learning-rate/">https://www.jeremyjordan.me/nn-learning-rate</a>
    </figcaption>
  </figure>
</div>

---

# Функция потерь
<div></div>

Функция потерь - это функция $L:(z,y)\ в \R \times Y \rightarrow L(z,y)\ в R$, которая принимает в качестве входных данных предсказанное значение $z$, соответствующее реальному значению данных $y$, и выдает, насколько они отличаются.
<div>
  <figure>
    <img src="/Loss_functions_cs-229.png" style="width: 750px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right: 60px; top: 60px"><br>Image source:
      <a href="https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning">by Shervine Amidi</a>
    </figcaption>
  </figure>
</div>

---

# Кросс-энтропия
<div></div>

<small>
  <small>

| Ширина пестика | Ширина тычинки | Вид    | $"p"$  | Кросс-энтропия      |
|-------------|-------------|------------|--------|--------------------|
| **0.04**        | **0.42**        | **Setosa**     |  **0.57**  | $\bm{-\mathrm{log}("p")}$ **= 0.56** |
| 1.0         | 0.54        | Virginica  |  0.58  | $-\mathrm{log}("p")$ = 0.54 |
| 0.50        | 0.37        | Versicolor |  0.52  | $-\mathrm{log}("p")$ = 0.65 |

</small>
</small>

<div>
  <figure>
    <img src="/Cross_entropy_1.svg" style="width: 700px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right: 60px; top: 60px"><br>Пример вдохновлен:
      <a href="https://www.youtube.com/watch?v=6ArSys5qHAU">Josh Starmer's video</a>
    </figcaption>
  </figure>
</div>

---

# Кросс-энтропия
<div></div>

<small>
  <small>

| Ширина пестика | Ширина тычинки | Вид    | $"p"$  | Кросс-энтропия      |
|-------------|-------------|------------|--------|--------------------|
| 0.04        | 0.42        | Setosa     |  0.57  | $-\mathrm{log}("p")$ = 0.56 |
| **1.0**         | **0.54**        | **Virginica**  |  **0.58**  | $\bm{-\mathrm{log}("p")}$ **= 0.54** |
| 0.50        | 0.37        | Versicolor |  0.52  | $-\mathrm{log}("p")$ = 0.65 |

</small>
</small>

<div>
  <figure>
    <img src="/Cross_entropy_2.svg" style="width: 700px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right: 60px; top: 60px"><br>Example inspired by:
      <a href="https://www.youtube.com/watch?v=6ArSys5qHAU">Josh Starmer's video</a>
    </figcaption>
  </figure>
</div>

---

# Кросс-энтропия
<div></div>

<small>
  <small>

| Ширина пестика | Ширина тычинки | Вид    | $"p"$  | Кросс-энтропия      |
|-------------|-------------|------------|--------|--------------------|
| 0.04        | 0.42        | Setosa     |  0.57  | $-\mathrm{log}("p")$ = 0.56 |
| 1.0         | 0.54        | Virginica  |  0.58  | $-\mathrm{log}("p")$ = 0.54 |
| **0.50**        | **0.37**        | **Versicolor** |  **0.52**  | $\bm{-\mathrm{log}("p")}$ **= 0.65** |

</small>
</small>

<div>
  <figure>
    <img src="/Cross_entropy_3.svg" style="width: 700px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right: 60px; top: 60px"><br>Example inspired by:
      <a href="https://www.youtube.com/watch?v=6ArSys5qHAU">Josh Starmer's video</a>
    </figcaption>
  </figure>
</div>

---

# Кросс-энтропия
<div></div>

<small>
  <small>

| Ширина пестика | Ширина тычинки | Вид    | $"p"$  | Кросс-энтропия      |
|-------------|-------------|------------|--------|--------------------|
| 0.04        | 0.42        | Setosa     |  0.57  | $\bm{-\mathrm{log}("p")}$ **= 0.56** |
| 1.0         | 0.54        | Virginica  |  0.58  | $\bm{-\mathrm{log}("p")}$ **= 0.54** |
| 0.50        | 0.37        | Versicolor |  0.52  | $\bm{-\mathrm{log}("p")}$ **= 0.65** |

</small>
</small>

<span style="margin-left: 400px;">Общаяя кросс-энтропия = 0.56 + 0.54 + 0.65 = 1.75</span>

<div>
  <figure>
    <img src="/Log_loss.png" style="width: 250px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Источник изображения:
      <a href="https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html">ml-cheatsheet.readthedocs.io</a>
    </figcaption>
  </figure>
</div>

---

# Использование популярных функций потерь
<div></div>

[Mean Absolute Error (MAE)](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss) Loss: $L(x, y) = |x - y|$

```python {all}
# MAE Loss
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
mae_loss = nn.L1Loss()
output = mae_loss(input, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(1.2850, grad_fn=<L1LossBackward>)
```

#### Когда она может быть использована?

* Регрессионные задачи. Считается, что MAE более устойчив к выбросам, чем RMSE.

<span style="color:grey"><small> Slides 11-17 are based on the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html) and on the [neptune.ai guide](https://neptune.ai/blog/pytorch-loss-functions).</small></span>
---

# Использование популярных функций потерь
<div></div>

[Mean Squared Error (MSE)](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) Loss: $L(x, y) = (x - y)^2$

```python {all}
# MSE Loss
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
mse_loss = nn.MSELoss()
output = mse_loss(input, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(2.3280, grad_fn=<MseLossBackward>)
```

#### Когда она может быть использована?

* Регрессионные задачи. MSE - функция потерь по умолчанию для большинства задач регрессии Pytorch.

---

# Использование популярных функций потерь
<div></div>

[Negative Log-Likelihood (NLL)](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) Loss: $L(x, y) = \{l_1,...,l_N\}^T$, where $l_N = -w_{y_n}x_{n,y_n}$. Softmax required!

```python {all}
# NLL Loss
import torch
import torch.nn as nn

# size of input (N x C) is = 3 x 5
input = torch.randn(3, 5, requires_grad=True)
# every element in target should have 0 <= value < C
target = torch.tensor([1, 0, 4])
m = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()
output = nll_loss(m(input), target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(2.9472, grad_fn=<NllLossBackward>)
```

#### Когда она может быть использована?

* Проблемы многоклассовой классификации

---

# Использование популярных функций потерь
<div></div>

[Cross Entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) Loss: $L(x, y) = -[y \cdot \mathrm{log}(x) + (1 - y) \cdot \mathrm{log}(1 - x)]$

```python {all}
# Cross Entropy Loss
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
cross_entropy_loss = nn.CrossEntropyLoss()
output = cross_entropy_loss(input, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(1.0393, grad_fn=<NllLossBackward>)
```

#### Когда она может быть использована?

* Задачи бинарной классификации (потери по умолчанию для классификации в PyTorch)

---
zoom: 0.94
---

# Использование популярных функций потерь
<div></div>

[Hinge Embedding](https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss) Loss: $L(x,y) = \begin{cases}
        x, \phantom{-1 <{}} \phantom{-1 <{}} \phantom{-1 <{}} \mathrm{\textcolor{grey}{if}~} y = 1 \\
        \mathrm{max}\{0, \Delta - x\}, \phantom{-1 <{}} \mathrm{\textcolor{grey}{~if}~} y = -1
      \end{cases}$

```python {all}
# Hinge Embedding Loss
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
hinge_loss = nn.HingeEmbeddingLoss()
output = hinge_loss(input, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(1.2183, grad_fn=<MeanBackward0>)
```

#### Когда она может быть использована?

* Задачи классификации, особенно при определении того, являются ли два входных сигнала несхожими или похожими
* Обучение нелинейным вкраплениям или задачи полуконтрольного обучения

---

# Использование популярных функций потерь
<div></div>

[Margin Ranking](https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss) Loss: $L(x_1, x_2, y) = \mathrm{max}(0, -y \cdot (x_1 - x_2) + \mathrm{margin})$

```python {all}
# Margin Ranking Loss
import torch
import torch.nn as nn

input_one = torch.randn(3, requires_grad=True)
input_two = torch.randn(3, requires_grad=True)
target = torch.randn(3).sign()

ranking_loss = nn.MarginRankingLoss()
output = ranking_loss(input_one, input_two, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(1.3324, grad_fn=<MeanBackward0>)
```

#### Когда она может быть использована?

* Проблемы ранжирования

---
zoom: 0.94
---

# Использование популярных функций потерь
<div></div>

[Kullback-Leibler Divergence (KLD)](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss) Loss: $L(x, y) = y\cdot(\mathrm{log}y - x)$

```python {all}
# Kullback-Leibler Divergence Loss
import torch
import torch.nn as nn

input = torch.randn(2, 3, requires_grad=True)
target = torch.randn(2, 3)
kl_loss = nn.KLDivLoss(reduction = 'batchmean')
output = kl_loss(input, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(0.8774, grad_fn=<DivBackward0>)
```

#### Когда она может быть использована?

* Аппроксимация сложных функций
* Задачи многоклассовой классификации
* Если вы хотите убедиться, что распределение предсказаний похоже на распределение обучающих данных