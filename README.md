# On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations
ReLU networks, while prevalent for visual data, have sharp transitions, sometimes relying on individual pixels for predictions, making vanilla gradient-based explanations noisy and difficult to interpret. 
Existing methods, such as GradCAM, smooth these explanations by producing surrogate models at the cost of faithfulness. 
We introduce a unifying spectral framework to systematically analyze and quantify smoothness, faithfulness, and their trade-off in explanations.

Using this framework, we quantify and reduce the contribution of ReLU networks to high-frequency information, providing a principled approach to identifying this trade-off. 
Our analysis characterizes how surrogate-based smoothing distorts explanations, leading to an ``explanation gap'' that we formally define and measure for different post-hoc methods.
Finally, we validate our theoretical findings across different design choices, datasets, and ablations.

---
this code base is somewhat improved compared to the ECCV24 paper.
