from typing import Literal, Optional

import jax
import jax.numpy as jnp
import equinox as eqx


class FourierFeatures(eqx.Module):
    B: Optional[jax.Array]
    frequencies: Optional[jax.Array]

    mode: Literal["gaussian", "dyadic"] = eqx.field(static=True)
    include_input: bool = eqx.field(static=True)
    trainable: bool = eqx.field(static=True)
    use_2pi: bool = eqx.field(static=True)
    encode_features: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        num_frequencies: int,
        key: Optional[jax.Array] = None,
        *,
        mode: Literal["gaussian", "dyadic"] = "gaussian",
        scale: float = 10.0,
        include_input: bool = True,
        trainable: bool = False,
        use_2pi: bool = True,
        encode_features: Optional[int] = None,
        dyadic_start: int = 1,
    ):
        """
        Fourier feature layer supporting either random Gaussian features or
        ordered dyadic features.

        Args:
            in_features:
                Total input dimensionality.

            num_frequencies:
                Number of Fourier frequencies.

                For mode="gaussian", this is the number of random projections.

                For mode="dyadic", this gives frequencies
                2^{dyadic_start}, ..., 2^{dyadic_start + num_frequencies - 1}.

            key:
                JAX PRNG key. Required if mode="gaussian".

            mode:
                Either "gaussian" or "dyadic".

            scale:
                Standard deviation of Gaussian matrix for mode="gaussian".

            include_input:
                If True, concatenate the original input to the Fourier features

            trainable:
                If True, Gaussian B or dyadic frequencies can receive gradients
                Usually False for fixed Fourier features.

            use_2pi:
                If True, use sin(2 pi omega x), cos(2 pi omega x).
                If False, use sin(omega x), cos(omega x).

                For the Geo-FNO-style formulation in your excerpt, use
                use_2pi=False.

            encode_features:
                Number of leading input coordinates to encode with Fourier
                features

                If None, encode all input features.

                This is useful for Geo-FNO-style inputs of the form [x, a],
                where x are spatial coordinates and a are geometry parameters.
                For example, if the input is [x1, x2, a1, a2, a3], then set
                encode_features=2 to apply sin/cos only to [x1, x2].

            dyadic_start:
                Starting exponent for dyadic frequencies. The default gives
                2^1, 2^2, ..., 2^K.
        """
        if encode_features is None:
            encode_features = in_features

        if encode_features > in_features:
            raise ValueError(
                f"encode_features={encode_features} cannot be larger than "
                f"in_features={in_features}."
            )

        self.mode = mode
        self.include_input = include_input
        self.trainable = trainable
        self.use_2pi = use_2pi
        self.encode_features = encode_features

        if mode == "gaussian":
            if key is None:
                raise ValueError("key must be provided when mode='gaussian'.")

            self.B = scale * jax.random.normal(
                key,
                shape=(encode_features, num_frequencies),
            )
            self.frequencies = None

        elif mode == "dyadic":
            self.B = None
            exponents = jnp.arange(
                dyadic_start,
                dyadic_start + num_frequencies,
            )
            self.frequencies = 2.0 ** exponents

        else:
            raise ValueError(f"Unknown Fourier feature mode: {mode}")

    @property
    def out_features(self) -> int:
        """
        Output feature dimension.
        """
        if self.mode == "gaussian":
            fourier_dim = 2 * self.B.shape[-1]
        elif self.mode == "dyadic":
            fourier_dim = 2 * self.frequencies.shape[0] * self.encode_features
        else:
            raise ValueError(f"Unknown Fourier feature mode: {self.mode}")

        if self.include_input:
            return self.encode_features + fourier_dim

        return fourier_dim

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x:
                Input array of shape (..., in_features)

        Returns:
            Fourier-encoded array.
        """
        x_encode = x[..., : self.encode_features]

        if self.mode == "gaussian":
            B = self.B if self.trainable else jax.lax.stop_gradient(self.B)

            xb = x_encode @ B

            if self.use_2pi:
                xb = 2.0 * jnp.pi * xb

            features = jnp.concatenate(
                [jnp.sin(xb), jnp.cos(xb)],
                axis=-1,
            )

        elif self.mode == "dyadic":
            frequencies = (
                self.frequencies
                if self.trainable
                else jax.lax.stop_gradient(self.frequencies)
            )

            xb = x_encode[..., None, :] * frequencies[:, None]

            if self.use_2pi:
                xb = 2.0 * jnp.pi * xb

            sin_xb = jnp.sin(xb)
            cos_xb = jnp.cos(xb)

            # Shape before reshape:
            #   (..., num_frequencies, encode_features)
            #
            # We concatenate as:
            #   [sin(2^1 x), cos(2^1 x),
            #    sin(2^2 x), cos(2^2 x), ...]
            features = jnp.concatenate([sin_xb, cos_xb], axis=-1)

            # Final shape:
            #   (..., num_frequencies * 2 * encode_features)
            features = features.reshape(*x.shape[:-1], -1)

        else:
            raise ValueError(f"Unknown Fourier feature mode: {self.mode}")

        if self.include_input:
            features = jnp.concatenate([x, features], axis=-1)

        return features
