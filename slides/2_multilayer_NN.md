---
zoom: 0.87
---

# Многослойные нейронные сети

<div class="grid grid-cols-[5fr_4fr]">
<div>

Рассмотрим нейронную сеть с 2 скрытыми слоями:
* **Первый скрытый слой** работает как в однослойной НС:<br>
$A_k^{(1)} = g(w_{k0}^{(1)} + \sum\limits_{j=1}^p w_{kj}^{(1)}X_j)$
* **Второй скрытый слой** обрабатывает активации<br>
из первого скрытого слоя:<br>
$A_l^{(2)} = g(w_{l0}^{(2)} + \sum\limits_{k=1}^{K_1} w_{lk}^{(2)}A_k^{(1)})$
* **Выходной слой**. Для $m = 0, 1, ..., 9$ нужно построить<br> 10 различных линейных моделей:<br> $Z_m = \beta_{m0} + \sum\limits_{l=1}^{K_2} \beta_{ml} A_l^{(2)}$
* Вероятность класса:<br>
$f_m (X) = \mathrm{Pr}(Y = m | X) = \frac{e^{Z_m}}{\sum_{l=0}^9 e^{Z_l}}$ (**softmax**)
</div>
<div>
  <figure>
    <img src="/ISLRv2_figure_10.4.png" style="width: 350px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 10px; position: absolute;"><br>НИ с двумя скрытыми слоями. Источник изображения:
      <a href="https://www.statlearning.com/">ISLP Рис. 10.4</a>
    </figcaption>
  </figure>
<br>

Обозначение:<br> $W_i$ - **веса** (коэффициенты), $B$ - **смещение**
</div>
</div>