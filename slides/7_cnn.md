---
layout: center
---
# Сверточные нейронные сети

---

# 2D сверточная нейронная сеть на MNIST <a href="https://adamharley.com/nn_vis/cnn/2d.html">[(ссылка)]</a>

<iframe src="https://adamharley.com/nn_vis/cnn/2d.html" width="1100" height="550" style="-webkit-transform:scale(0.8);-moz-transform-scale(0.8); position: relative; top: -65px; left: -120px"></iframe>

---

# Сверточные нейронные сети (CNN)
### CNN представляет собой последовательность сверточных слоев, перемежающихся функциями активации, пулингом и регуляризацией
<br>
<br>
<div>
  <figure><center>
    <img src="/cnn_layers.png" style="width: 700px !important;">
</center>
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Источник изображения:
    	<a href="http://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf">Andrej Karpathy</a>
    </figcaption>
  </figure>   
</div>

---
zoom: 0.94
---

# Архитектура CNN
#### Общая архитектура CNN может быть представлена в виде:
<br>

### Вход
<br>
<br>

### Конволюционные блоки
* Свертка + Активация (ReLU)
* Свертка + Активация (ReLU)
* ...
* Maxpooling
<br>
<br>

### Выходные данные
* Полносвязанные слои
* Softmax

---
zoom: 0.94
---

# CNN для глубокого обучения
### **Глубокое обучение** = обучение иерархическим представлениям
#### Глубокое обучение - это обучение сети, в которой имеется более одного этапа нелинейного преобразования признаков
<br>
<br>
<div>
  <figure><center>
    <img src="/cnn_hierarchical_representation.png" style="width: 500px !important;">
</center>
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Источник изображения
    	<a href="https://drive.google.com/file/d/18UFaOGNKKKO5TYnSxr2b8dryI-PgZQmC/view?usp=share_link">Yann LeCun</a>
    </figcaption>
  </figure>   
</div>

---

# Сверточные нейронные сети
<br>
<div>
  <figure><center>
    <img src="/cnn_architecture.jpg" style="width: 700px !important;">
</center>
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Источник изображения:
    	<a href="http://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf">Andrej Karpathy</a>
    </figcaption>
  </figure>   
</div>

---
zoom: 0.94
---

# Современное состояние

### Поиск правильных архитектур: Активная область исследований


<div>
  <figure><center>
    <img src="/cnn_sota_1.png" style="width: 550px !important;">
</center>
  </figure>   
</div>
<br>
<br>

#### Модульное проектирование строительных блоков сети

#### См. также: Плотные сети, Широкие сети, Фрактальные сети, Сети ResNeXts, Пирамидальные сети

<span style="color:grey"><small>From Kaiming He slides "Deep residual learning for image recognition." ICML. (2016)</small></span>

---

# Современное состояние

### Top 1-accuracy, производительность и размер сетей на ImageNet

<div>
  <figure><center>
    <img src="/cnn_sota_2.png" style="width: 750px !important;">
</center>
  </figure>   
</div>
<br>
<br>

#### См. также: https://paperswithcode.com/sota/image-classification-on-imagenet

<span style="color:grey"><small>From Canziani, Paszke, and Culurciello. "An Analysis of Deep Neural Network Models for Practical Applications." (May 2016)</small></span>
