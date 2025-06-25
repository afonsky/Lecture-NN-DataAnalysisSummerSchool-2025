---
zoom: 0.87
---

# Подгонка нейронной сети

<div class="grid grid-cols-[5fr_2fr]">
<div>

* $\theta$ - параметры модели:<br>
$\beta = (\beta_0, \beta_1, ..., \beta_K)$ и $w_k = (w_{k0}, w_{k1}, ..., w_{kp})$<br>
* Нужно решить нелинейную задачу наименьших квадратов:<br>
$\underset{\{w_k\}_1^K, \beta}{\mathrm{minimize}} \frac{1}{2}\sum\limits_{i=1}^n (y_i - f(x_i))^2$,

где $f(x_i) = \beta_0 + \sum\limits_{k=1}^K \beta_k g(w_{k0} + \sum\limits_{j=1}^p w_{kj}x_{ij})$

#### Задача **невыпуклая** по параметрам $\leadsto$, что означает наличие множества решений.
</div>
<div>
  <figure>
    <img src="/ISLRv2_figure_10.1.png" style="width: 250px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 10px; position: absolute;"><br>Полносвязная нейронная сеть<br>Источник изображения:
      <a href="https://www.statlearning.com/">ISLP Рис. 10.1</a>
    </figcaption>
  </figure>
</div>
</div>

<div class="grid grid-cols-[3fr_5fr]">
<div>
  <figure>
    <img src="/ISLRv2_figure_10.17.png" style="width: 250px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 10px; position: absolute;"><br>Градиентный спуск для одномерной θ. Источник изображения:
      <a href="https://www.statlearning.com/">ISLP Рис. 10.17</a>
    </figcaption>
  </figure>
</div>

<div>

Чтобы преодолеть некоторые из этих проблем, мы можем использовать:
* **Медленное обучение**
  * **Градиентный спуск**
* **Регуляризация** (Ridge/Lasso)
</div>
</div>

---
zoom: 0.87
---

# Подгонка нейронной сети: Градиентный спуск

<div class="grid grid-cols-[3fr_5fr]">
<div>
  <figure>
    <img src="/ISLRv2_figure_10.17.png" style="width: 250px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Градиентный спуск для одномерной θ.<br> Источник изображения:
      <a href="https://www.statlearning.com/">ISLP Рис. 10.17</a>
    </figcaption>
  </figure>
<br>
<br>
<br>

  <figure>
    <img src="/Gradient_Descent.png" style="width: 350px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Источник изображения:
      <a href="https://easyai.tech/en/ai-definition/gradient-descent/">https://easyai.tech/en/ai-definition/gradient-descent</a>
    </figcaption>
  </figure>
</div>

<div>

Переписав задачу наименьших квадратов в виде:<br>
$R(\theta) = \frac{1}{2}\sum\limits_{i=1}^n (y_i - f_\theta(x_i))^2$.

можно сформулировать общий алгоритм **градиентного спуска**:
1. Начнем с предположения $\theta^0$ для всех параметров в $\theta$ и зададим $t = 0$.
2. Итерируем до тех пор, пока целевое значение $R(\theta)$ **не перестанет уменьшаться**:
  1. Найдем вектор $\delta$, отражающий небольшое изменение $\theta$, такое, что<br> $\theta^{t+1}$ = $\theta^t + \delta$ уменьшает $R(\theta)$;<br> т.е. такое, что $R(\theta^{t+1}) < R(\theta^t)$.
  2. Зададим $t \leftarrow t + 1$
</div>
</div>
<br>

---

# Как выбрать $\theta^0$?

#### Распространенные инициализации весов: [Xavier](https://proceedings.mlr.press/v9/glorot10a), [Kaiming (He)](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
<br>

<figure>
    <img src="/adamw_optimization_trajectory.png" style="width: 855px !important;">
        <figcaption style="color:#b3b3b3ff; font-size: 10px; position: absolute"><br>Источник изображения:
      <a href="https://www.mdrk.io/optimizers-in-deep-learning/">https://mdrk.io/optimizers-in-deep-learning</a>
    </figcaption>
</figure>

<br>
<br>

##### Взгляните на работу Анкура Мохана [интерактивный 3D-визуализатор поверхностей функции потерь](https://www.telesens.co/loss-landscape-viz/viewer.html)
