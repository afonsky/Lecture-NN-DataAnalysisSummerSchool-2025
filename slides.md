---
# You can also start simply with 'default'
theme: seriph
addons:
  - "@twitwi/slidev-addon-ultracharger"
addonsConfig:
  ultracharger:
    inlineSvg:
      markersWorkaround: false
    disable:
      - metaFooter
      - tocFooter

background: /mountain.jpg

# some information about your slides (markdown enabled)
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
# transition: slide-down
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true

title: Нейронные сети
hideInToc: true
date: 25/06/2025
venue: ФКН ВШЭ
author: Алексей Болдырев
---

<br>
<br>
<br>
<br>

# Нейронные сети<br><br>
### Алексей Болдырев<br><br>
#### Летняя школа по анализу данных<br>
#### ФКН ВШЭ 25/06/2025
<div>
<br>
<span style="color:#b3b3b3ff; font-size: 11px; float: right;">Изображение: ‘Glacier du Rhone au haut du Valais’<br> by Claude Niquet after Jean Séraphin Désiré Besson<br>
<a href="https://wellcomecollection.org/works/e3y95vtv">https://wellcomecollection.org/works/e3y95vtv</a>
</span>
</div>

<style>
  :deep(footer) { padding-bottom: 3em !important; }
</style>

---
src: ./slides/0_introduction.md
---

---
src: ./slides/1_single_layer_NN.md
---

---
src: ./slides/2_multilayer_NN.md
---

---
src: ./slides/3_fitting_NN.md
---

---
src: ./slides/4_backpropagation.md
---

---
src: ./slides/5_convolution_layer.md
---

---
src: ./slides/6_pooling_layer.md
---

---
src: ./slides/7_cnn.md
---

---
src: ./slides/9_DL_tools.md
---

---
src: ./slides/0_supplementary.md
---

---
src: ./slides/0_end.md
---
