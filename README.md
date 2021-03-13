---
typora-copy-images-to: ./
---

# surface visualization

Visualize the loss surface. This project is a shallow implementation of "Visualizing the Loss Landscape of Neural Nets."

![image-20210313182606596](image-20210313182606596.png)

direction vector is a random vector sampled from the normal distribution. The direction vector D is obtained as follows.

```python
direction = torch.randn(param.size()) * param
```

Bias and normalization were excluded from direction calculation (Set to 0)

![image-20210313182546417](image-20210313182546417.png)

![image-20210313182531243](image-20210313182531243.png)

![image-20210313182500231](image-20210313182500231.png)

![image-20210313182358058](image-20210313182358058.png)



