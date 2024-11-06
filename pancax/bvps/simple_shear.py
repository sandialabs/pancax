from typing import Optional

def SimpleShearLinearRamp(
  final_displacement: float,
  length: float,
  direction: str
):
    
  if direction == 'xy':
    def bc_func(xs, t, z):
      x, y = xs[0], xs[1]
      u_out = z
      u_out = u_out.at[0].set(
        x * (x - length) * t * z[0] / length**2
      )
      u_out = u_out.at[1].set(
        x * t * final_displacement / length + x * (x - length) * t * z[1] / length**2
      )
      return u_out
  else:
    raise ValueError('Direction must be x or y for this BVP.')

  return bc_func