# Forward Diffusion Equation - Single Step

![Forward Diffusion Single Step](../images_for_slides/single_step.png)

The equation below describes a single step in the **forward diffusion process**. This is the core mechanism for gradually adding noise to an image until it becomes unrecognizable.

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

Here's how to read and understand each part of it:

### The Distribution: Normal (or Gaussian)
The **$\mathcal{N}$** in the equation signifies that we are sampling from a **Normal Distribution**, also known as a Gaussian distribution. This is a very common probability distribution that produces the classic "bell curve" shape. In this context, it's used as the mathematical model for the random noise that is added to the image at each step.

### The Left Side: $q(\mathbf{x}_t | \mathbf{x}_{t-1})$
This part of the equation describes the probability of arriving at the state of the image at the current timestep $t$ (which is $\mathbf{x}_t$), given the image from the previous timestep $t-1$ (which is $\mathbf{x}_{t-1}$). You can think of it as the formal rule for taking one small step forward in the noise-adding process.

### The Parameters (Inside the $\mathcal{N}$)
A normal distribution is defined by two primary parameters: its mean (the center of the bell curve) and its variance (which controls how spread out the curve is).

1.  **Mean (The Center):** $\sqrt{1 - \beta_t} \mathbf{x}_{t-1}$
    *   This term represents the **mean** of the distribution.
    *   It's calculated by taking the image from the previous step ($\mathbf{x}_{t-1}$) and scaling it down by a factor of $\sqrt{1 - \beta_t}$. Since $\beta_t$ is a small positive number, this factor is slightly less than 1.
    *   This exact operation is reflected in the provided Python code: `mean = np.sqrt(1 - beta_t) * x_prev`.

2.  **Variance (The Spread):** $\beta_t \mathbf{I}$
    *   This term is the **variance**, and it controls the amount and nature of the noise being added at this step.
    *   $\beta_t$ (beta_t): This is a small, positive hyperparameter that determines the *strength* or *amount* of noise added at step $t$. As noted in the image, $\beta_t$ is gradually increased over time according to a "noise schedule." This ensures that noise is added gently at first and more aggressively in later steps.
    *   $\mathbf{I}$ (Identity Matrix): This signifies that the noise is applied independently to each pixel of the image with the same variance ($\beta_t$).

In simple terms, each forward step takes the previous image, makes it slightly fainter, and then adds a specific amount of new, random noise to every pixel to create the next image in the sequence.