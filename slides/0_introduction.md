---
zoom: 0.9
---

# Биологические и искусственные нейронные сети

<div class="grid grid-cols-[5fr_2fr]">
<div>
  <figure>
    <img src="/Neuron3.svg" style="width: 500px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 10px; position: absolute;">Источник изображения:
      <a href="https://commons.wikimedia.org/wiki/File:Neuron3.svg">https://commons.wikimedia.org/wiki/File:Neuron3.svg</a>
    </figcaption>
  </figure>
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

<div class="grid grid-cols-[1fr_1fr]">
<div>
<br>
<v-clicks>

* <small>Обе имеют *много* **входов** от и **выходов** к другим нейронам</small>
* <small>В обоих случаях используется **активация** нейронов</small>
* <small>Обе **предназначены для обучения** оптимальному поведению</small>

</v-clicks>
</div>
<div>
<v-clicks>
<small>В искусственных NN:</small>

* <small>"**дендриты**" - это связи, которые несут информацию<br> (усвоенные коэффициенты).</small>
* <small>"**синапсы**" - это функции активации, которые дополняют или фильтруют поток информации; а "**сома**" выступает в качестве функции суммирования</small>

</v-clicks>
</div>
</div>

---
zoom: 0.9
---

# Биологические и искусственные нейронные сети

<div class="grid grid-cols-[5fr_2fr]">
<div>
  <figure>
    <img src="/Neuron3.svg" style="width: 500px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 10px; position: absolute;">Источник изображения:
      <a href="https://commons.wikimedia.org/wiki/File:Neuron3.svg">https://commons.wikimedia.org/wiki/File:Neuron3.svg</a>
    </figcaption>
  </figure>
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

<div class="grid grid-cols-[3fr_2fr]">
<div>
<br>
<small>Дополнительная литература по биологическим нейронным сетям: <a href="https://christofkoch.com">Christof Koch:</a></small>

* <small><a href="https://christofkoch.com/biophysics-book/">Biophysics of Computation: Information Processing in Single Neurons
</a></small>
* <small><a href="https://www.cse.psu.edu/~rtc12/CSE597E/papers/Itti_etal98pami.pdf">A model of saliency-based visual attention for rapid scene analysis</a></small>
* <small><a href="https://www.youtube.com/watch?v=indbWawx3Hs">Consciousness & Reality Colloquium Series: Inaugural Lecture</a></small>
</div>
<div>

* <small>Neuroscience by Dale Purves et al. (6th ed., 2018)</small><br>
<small>В. Дубынин:</small>
* <small>Мозг и его потребности: От питания до признания (2021)</small>
* <small><a href="https://www.youtube.com/@dubynin/playlists">Курс лекций (видео)</a></small>
</div>
</div>

---
zoom: 0.9
---

# Биологические и искусственные нейронные сети
<div>
</div>

Искусственные нейронные сети **вдохновлены** биологическими нейронными сетями<br> но большинство из них лишь **отдаленно напоминают** последние.<br>

<a href="https://ru.wikipedia.org/wiki/%D0%98%D0%BC%D0%BF%D1%83%D0%BB%D1%8C%D1%81%D0%BD%D0%B0%D1%8F_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C">Импульсные нейронные сети</a> наиболее точно имитируют естественные нейронные сети.

<div class="grid grid-cols-[5fr_3fr]">
<div>
  <figure>
    <img src="/Unsupervised_learning_with_ferroelectric_synapses.png" style="width: 490px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 10px; position: absolute;"><br>Обучение без учителя с помощью ферроэлектрических синапсов.<br> Источник изображения:
      <a href="https://www.nature.com/articles/ncomms14736"><em>Nature Communications</em> 8, 14736 (2017)</a>
    </figcaption>
  </figure>
</div>
<div>

#### [Brain Score](http://www.brain-score.org)
<br>
  <figure>
    <img src="/gr3_lrg.jpg" style="width: 400px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 10px; position: absolute;"><br>Бенчмарк для сравнения нейромеханистических моделей человеческого интеллекта. Источник изображения:
      <a href="https://doi.org/10.1016/j.neuron.2020.07.040"><em>Neuron</em> 108.3 (2020)</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Искусственные нейронные сети: Обзор

<v-clicks>

* [ИНС](https://ru.wikipedia.org/wiki/%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C) - это гибкий класс моделей, который может находить сильно нелинейные отношения в системах ввода-вывода
* ИНС - это старая технология, возрожденная благодаря недавнему буму GPU/TPU, новым алгоритмам, получению прибыли и крупным инвесторам
* ИНС строится из нейронов, основных строительных блоков
* ИНС помогает сконцентрировать усилия на инженерной инфраструктуре, а не на подстройке признаков
* ИНС более эффективны (по сравнению с классическим ML) в задачах с неструктурированными данными: текстом, аудио, изображениями, видео, ...
* **ИНС охватывает множество инфраструктур**

</v-clicks>

---
zoom: 0.9
---

# Искусственные нейронные сети: Примеры

<v-clicks>

1. [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) для последовательных данных с короткими зависимостями
1. [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) для последовательных данных с короткими и долгими зависимостями
1. [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) для изображений с двухмерными и трехмерными пространственными зависимостями
1. [U-Net](https://en.wikipedia.org/wiki/U-Net) для сегментации изображений и не только
1. [VAE](https://en.wikipedia.org/wiki/Variational_autoencoder) для сжатия изображений, аудио, ...
1. [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network) для генерации новых наблюдений (например, лиц, голосов) из обучающего распределения
1. [Transformers](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) создает "*внимание*" к "*важным*" входным данным
   * [LLMs](https://en.wikipedia.org/wiki/Large_language_model) использует архитектуру трансформатора
1. [Deep RL](https://en.wikipedia.org/wiki/Deep_reinforcement_learning) Обучение агента действиям с максимальным вознаграждением на основе текущего состояния и прошлой истории (например, игры, робототехника)
1. [GNN](https://en.wikipedia.org/wiki/Graph_neural_network) для данных на основе графов (например, социальные сети, карты улиц)
1. [RBM](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) для изучения распределения входных данных для генеративных задач
1. [SOM](https://en.wikipedia.org/wiki/Self-organizing_map) для уменьшения размерности с сохранением топологической структуры

</v-clicks>

---
zoom: 0.9
---

# Основы искусственных нейронных сетей

### Строительные блоки:
<div class="grid grid-cols-[3fr_2fr_2fr] gap-3">
<div>

* Нейрон
* Функция потерь
* Функция активации
* Оптимизатор
<!-- * <span style="color:#FA9370">Optimizer</span> -->
</div>

<div>

* Линейный слой
* Свёрточный слой
* Слой пулинга
* Рекуррентный слой
* Слой внимания
</div>

<div>
  <figure>
    <img src="/lego_A.jpg" style="width: 190px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Источник изображения:
      <a href="http://sgaguilarmjargueso.blogspot.com/2014/08/de-lego.html">http://sgaguilarmjargueso.blogspot.com</a>
    </figcaption>
  </figure>   
</div>
</div>
<br>

### Концепции:
<div class="grid grid-cols-[2fr_2fr_3fr] gap-2">
<div>

* Веса и смещения
* Обратное<br> распространение
* Градиентный спуск

</div>
<div>

* Скорость обучения
* Мини-батч
* Регуляризация
</div>

<div>

* Длительная краткосрочная память (LSTM)
* Кодировщик и декодировщик
</div>

</div>
