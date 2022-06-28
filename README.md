# Centernet-MDN
Original Paper: [CenterNet](https://arxiv.org/abs/1904.07850),  [MDN](https://arxiv.org/pdf/1709.02249)

Original Code From [code](https://github.com/xingyizhou/CenterNet)


Mixture Density Network + CenterNet

**Architecture**
<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/Centernet-MDN/blob/main/framework.png">
</p>

**Loss**
<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/Centernet-MDN/blob/main/loss.png">
</p>

### Qualitative Results

<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/Centernet-MDN/blob/main/mln_centernet.png">
</p>

Epistemic uncertainty is high on the edge of the object, Aleatoric uncertainty is high on the center of the object.

### Run Code

```
python main.py
```
