import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Device types: {[d.device_kind for d in jax.devices()]}")

# 测试GPU计算
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
result = x + y
print(f"GPU computation test: {result}")