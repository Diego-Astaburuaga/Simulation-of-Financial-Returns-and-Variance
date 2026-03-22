"""ML generative models for return simulation (placeholder).

Planned functionality
---------------------
* Variational Autoencoder (VAE)

  - ``train_vae(returns, latent_dim=8, epochs=200, batch_size=64)``
      Train a VAE on a sliding-window return matrix and return the fitted model.
  - ``sample_vae(model, n_paths, n_steps)``
      Draw samples from the VAE latent space and decode them to return paths.

* Generative Adversarial Network (GAN) / TimeGAN

  - ``train_gan(returns, latent_dim=16, epochs=500, batch_size=64)``
      Train a GAN (generator + discriminator) on the historical return series.
  - ``sample_gan(generator, n_paths, n_steps)``
      Generate synthetic return paths from the trained generator.

Suggested dependencies
----------------------
- PyTorch or TensorFlow / Keras.
- ``tsgan`` or custom TimeGAN implementation.

Notes
-----
ML models are data-hungry.  It is recommended to use at least 5 years of
daily data (≈1 250 observations) before training these models.

References
----------
- Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
- Yoon et al. (2019) "Time-series Generative Adversarial Networks" (NeurIPS)
"""

# TODO: implement train_vae / sample_vae
# TODO: implement train_gan / sample_gan
