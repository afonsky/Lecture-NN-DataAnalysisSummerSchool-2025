# <center>Нейронные сети</center>
<br>
<br>

<div>
<center>
  <figure>
    <img src="/nn_patterns_1.png" style="width: 550px !important;">
  </figure>
</center>
</div>
<br>
<br>

# <center>Умеют распознавать закономерности в данных (например, в цифрах)</center>
<br>
<br>
<br>
<span style="color:grey"><small>Slides 20-26 are adapted from <a href="https://deeplearning.cs.cmu.edu/F22/document/slides/lec9.CNN1.pdf">Bhiksha Raj's slides</a></small></span>

---

# <center>Веса помогают находить закономерности</center>

<br>

<div>
<center>
  <figure>
    <img src="/nn_patterns_2.png" style="width: 620px !important;">
  </figure>
</center>   
</div>
<br>
<br>

### Зеленый паттерн больше похож на паттерн весов (черный), чем на красный паттерн
* Зеленый паттерн более *скоррелирован* с паттерном весов

---

# <center>А что будет с цветком?</center>
<br>

<div>
<center>
  <figure>
    <img src="/nn_patterns_3.jpg" style="width: 650px !important;">
  </figure>
</center>    
</div>
<br>
<br>

# <center>Есть ли на этих изображениях цветок?</center>

---

# <center>Цветок</center>
<br>

<div>
<center>
  <figure>
    <img src="/nn_patterns_4.jpg" style="width: 650px !important;">
  </figure>
</center>   
</div>
<br>

* Сможет ли сеть, распознающая левое изображение как цветок, также распознавать цветок на правом изображении?
* Нужна сеть, которая будет работать независимо от точного местоположения целевого объекта

---

# Нам необходима инвариантность относительно сдвига
<br>

* Во многих задачах расположение детали не имеет значения
  * Важно только присутствие паттерна
* Обычные нейронные сети чувствительны к расположению паттерна
  * Перемещение его на один компонент приводит к совершенно другому
вводу, который нейронная сеть не распознает
* Требование: Сеть должна быть **инвариантной ко сдвигам**

---

# Решение: сканирование

<div>
<center>
  <figure>
    <img src="/nn_patterns_5.jpg" style="width: 600px !important;">
  </figure>
</center>   
</div>

### Сканирование в поисках нужного объекта
* Будем "искать" целевой объект в каждом положении
* В каждом месте вся область отправляется через нейронную сеть

---
zoom: 0.95
---

# Решение: сканирование

<div>
<center>
  <figure>
    <img src="/nn_patterns_6.jpg" style="width: 550px !important;">
  </figure>
</center>   
</div>

### Определим, есть ли в любом из этих мест цветок
* Каждый нейрон в правой части представляет собой выход нейронной сети, когда он классифицирует одно местоположение во входном изображении
* Посмотрим на максимальное значение
  * Или пропустим его через простую нейронную сеть (например, из линейных слоев и softmax активации)

---

# Сверточный слой

### Этот подход, похожий на сканирование, реализован в **сверточных слоях** нейронных сетей:

<div>
<center>
  <figure>
    <img src="/conv_2D_1.gif" style="width: 550px !important;">
  </figure>
</center>   
</div>

* Голубая (нижняя) матрица - это $5\times 5$ **вход**
* Синяя (тень) - это $3\times 3$ **ядро** (оно же **фильтр** или **окно**)
* Зеленая (верхняя) матрица - $3\times 3$ **выход** (она же **карта признаков** или **карта активации**)


<br>
<br>
<span style="color:grey"><small>Gifs from <a href="https://arxiv.org/abs/1603.07285">A guide to convolution arithmetic for deep learning</a> by Vincent Dumoulin and Francesco Visin</small></span>

---
zoom: 0.95
---

# Сверточный слой (1D)
<div>
<center>
  <figure>
    <img src="/conv_1D_1.gif" style="width: 350px !important;">
  </figure>
</center>   
</div>

#### Свертка - это операция над двумя сигналами: входным и ядром.

#### Чтобы получить результат свертки входного вектора и ядра:
- **Пройдитесь ядром** по всевозможным позициям входного вектора
- Для каждой позиции выполните **элементное произведение** между ядром и соответствующей частью входных данных
- **Просуммируйте** результат поэлементного произведения

<span style="color:grey"><small>Гифки из <a href="https://arxiv.org/abs/1603.07285">A guide to convolution arithmetic for deep learning</a> by Vincent Dumoulin and Francesco Visin</small></span>

---
zoom: 0.94
---

# Сверточный слой
<div></div>

Предполагая заданный вход $I$ с учетом его размерности, мы можем выбрать следующие
**гиперпараметры** сверточного слоя:
* размер ядра (**фильтра**) $F$
* **Stride** $S$ - количество *пикселей*, на которое перемещается фильтр после каждой операции
* **Zero-padding** $P$ - количество нулей за границами входного сигнала

<div class="grid grid-cols-[5fr_2fr]">
<div>
<center>
  <figure>
    <img src="/conv_layer_1.png" style="width: 450px !important;">
  </figure>
</center>   
</div>
<div>
<br>
<br>
<br>
Тогда:

  $\boxed{O = \frac{I - F + 2P}{S} + 1}$
</div>
</div>

<br>
<br>
<span style="color:grey"><small>Источник изображений: <a href="https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks">Convolutional Neural Networks cheatsheet</a></small></span>

---
zoom: 0.94
---

# Сверточный слой
<div></div>

Можно использовать *много каналов* на входе. Цветное изображение обычно имеет 3 входных канала: RGB (красный, зеленый, синий).
Поэтому ядро также будет иметь каналы, по одному на каждый входной канал.

<div class="grid grid-cols-[5fr_3fr]">
<div>
  <figure>
    <img src="/conv-2d-in-channels.gif" style="width: 550px !important;">
  </figure>  
</div>
<div>
  <figure>
    <img src="/conv-2d-out-channels.gif" style="width: 400px !important;">
  </figure>
</div>
</div>

<br>
<span style="color:grey"><small>Гифки из <a href="https://github.com/theevann/amld-pytorch-workshop/blob/master/6-CNN.ipynb">the PyTorch Workshop at Applied ML Days 2019</a></small></span>
