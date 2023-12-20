# Summary

- We conducted a series of experiments on the MNIST dataset, including  ddpm and conditional ddpm, as well as cold diffusion and score based model.
- You can run the main.py with command("ddpm/ddpm_conditional/cold_median/cold_kernel/cold_resolution/score") to see our results of experiments.
- You can see the eval_score_matching.ipynb for the calculation of the energy function with score matching to evaluate the results of all the experiments above.

## DDPM and DDPM Condtional

We use U-NET to generate a new image based on the image after adding noise

![](figure/conditional.jpg)

![](figure/DDPM.jpg)

## Cold Diffusion

We tried three deterministic operations, adding median blur, mean blur and masking part of the image.

### Median 
![](figure/median.jpg)

### Kernel 
![](figure/kernel.jpg)

### Super-resolution 
![](figure/resolution.jpg)

## Score Based Model

Following the idea of [], we construct a Noise Conditional Score Networks and apply it to Mnist dataset, with the Annealed Langevin Dynamic sampling method, we can generate new images as shown in `/results`. 

![](figure/Score.jpg)

## Energy function to evaluate the images


![](figure/Energy.jpg)


## References

* A Connection Between Score Matching and Denoising Autoencoders, https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf
* Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239
* Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise,https://arxiv.org/abs/2208.09392
* Generative Modeling by Estimating Gradients of the Data Distribution,https://yang-song.net/blog/2021/score/
