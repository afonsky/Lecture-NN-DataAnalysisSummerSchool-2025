---
zoom: 0.94
---

# Слой пулинга
<div></div>

Слой объединения (**POOL**) - это операция понижения дискретизации, обычно применяемая после слоя свертки, которая обеспечивает некоторую пространственную инвариантность. Наиболее популярные виды пулинга:
<div class="grid grid-cols-[1fr_1fr]">
<div>

* **Max pooling**
  * Выбирает максимальное значение текущего вида
  * Сохранение обнаруженных особенностей

```python {all}
m = nn.MaxPool2d(2, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
```
</div>
<div>
  <figure>
    <img src="/max-pooling-a.png" style="width: 200px !important;">
  </figure>  
</div>
</div>

<div class="grid grid-cols-[1fr_1fr]">
<div>

* **Average pooling**
  * Усредняет значения текущего представления
  * Уменьшение выборки карты признаков
  * Используется в LeNet
</div>
<div>
  <figure>
    <img src="/average-pooling-a.png" style="width: 200px !important;">
  </figure>
</div>
</div>
<span style="color:grey"><small>Гифки из <a href="https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks">Convolutional Neural Networks cheatsheet</a></small></span>