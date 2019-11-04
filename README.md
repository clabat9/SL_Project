# A model for speakers and places recognition

![](https://nikegroup.it/wp-content/uploads/2017/12/artificial-intelligence.jpg)

With this work we propose a model to recognize the speaker and the place of registrations token using **Science Journal** app. These registrations are made of various kinds of environment sensing implemented through the mobile's sensors.  We faced the task using most of the main tools studied during the **Statistical Learning** course; in particular, we proceeded with some fundamental and essential guidelines:

- Sensing is, obviously, time dependent and we have to take this into account, so the basis of the work is **Functional Analysis**, in particular basis expansion.

- Basis expansion means, in general, that there will be a huge number of features (coefficients) in the dataset, so another keyword is **Dimensionality Reduction**.

- ML problems and techniques are constantly evolving and we thought a good idea is using **State-of-the-Art** methods. In particular, here we propose a graph-based semi-supervised clustering algorithm. The paper poster can be freely read [here](https://sigport.org/sites/default/files/docs/PosterGlobalSip_2.pdf).

All the detailed descriptions of scripts and logic are in **report.html** (not readable on GitHub, you can just use a viewer.
