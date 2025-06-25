# Обратное распространение

#### Как найти направление перемещения $\theta$, чтобы уменьшить цель $R(\theta)$?

Нужно вычислить **градиент** $R(\theta)$, оцененный при некотором текущем значении $\theta = \theta^m$:

$\nabla R(\theta^m) = \frac{\partial R(\theta)}{\partial\theta} \biggr\rvert_{\theta = \theta^m}$

Идея градиентного спуска состоит в том, чтобы немного сдвинуть $\theta$ в обратном направлении:

$\theta^{m+1} \leftarrow \theta^m - \rho \nabla R(\theta^m)$,

где $\rho$ - **скорость обучения**.

Если вектор градиента равен нулю, то, возможно, мы достигли минимума цели.

#### **Обратное распространение**:
* Позволяет вычислять градиенты алгоритмически
* Используется во фреймворках глубокого обучения (TensorFlow, PyTorch и т. д.)

---

# Обратное распространение: пример

<div class="grid grid-cols-[2fr_2fr]">
<div>
  <figure>
    <img src="/Backpropagation_Ex2.png" style="width: 350px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Источник:
      <a href="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/syllabus.html">https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184</a>
    </figcaption>
  </figure>
</div>
<div>

$$
\boxed{
\begin{array}{rcl}
f(x, y, z) = (x + y) ~\mathrm{max}(y, z)\\
x = 1, ~y = 2, ~z = 0\\
\\
\mathrm{Найдем~}\frac{\partial{f}}{\partial{y}}
\end{array}
}
$$
</div>
</div>

<div class="grid grid-cols-[2fr_2fr]">
<div>

Прямое распространение:<br>
$a = x + y$

$b = \mathrm{max}(y, z)$

$f = a \cdot b$

</div>
<div>

Локальные градиенты:<br>
$\frac{\partial{a}}{\partial{x}} = 1$ $~~~\frac{\partial{a}}{\partial{x}} = 1$

$\frac{\partial{b}}{\partial{y}} = \boldsymbol{1} (y > z) = 1$ $~~~\frac{\partial{b}}{\partial{z}} = \boldsymbol{1} (y < z) = 0$

$$
\boxed{
\frac{\partial{f}}{\partial{a}} = b = 2~~~ \frac{\partial{f}}{\partial{b}} = a = 3
}
$$
</div>
</div>

$\frac{\partial{f}}{\partial{x}} = 2~~~~~~ \frac{\partial{f}}{\partial{y}} = 3 + 2 = 5~~~~~ \frac{\partial{f}}{\partial{z}} = 0$

---

# Обратное распространение в PyTorch

```python {all}
# Инициализируйте x, y и z значениями 4, -3 и 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5., requires_grad=True)

# Задайте q суммой x и y, а f - произведением q на z
q = x + y
f = q * z

# Вычислите производные
f.backward()

# Выведем на экран градиенты
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))
```
Приведенный выше код выдает:
```python {all}
Gradient of x is: tensor(5.)
Gradient of y is: tensor(5.)
Gradient of z is: tensor(1.)
```
