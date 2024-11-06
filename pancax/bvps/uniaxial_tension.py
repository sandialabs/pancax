
# TODO maybe have this read in a mesh
# and use a bounding box to define these functions using xmin xmax etc.
def UniaxialTensionLinearRamp(
  final_displacement: float,
  length: float,
  direction: str,
  n_dimensions: int
):

  if n_dimensions == 2:
    if direction == 'x':
      def bc_func(xs, t, nn):
        x, y = xs[0], xs[1]
        u_out = nn
        u_out = u_out.at[0].set(
          x * t * final_displacement / length + x * (x - length) * t * nn[0] / length**2
        )
        u_out = u_out.at[1].set(
          x * (x - length) * t * nn[1] / length**2
        )
        return u_out
    elif direction == 'y':
      def bc_func(xs, t, nn):
        x, y = xs[0], xs[1]
        u_out = nn
        u_out = u_out.at[0].set(
          y * (y - length) * t * nn[0] / length**2
        )
        u_out = u_out.at[1].set(
          y * t * final_displacement / length + y * (y - length) * t * nn[1] / length**2
        )
        return u_out
    else:
      raise ValueError('Direction must be x or y for this BVP.')
  elif n_dimensions == 3:
    if direction == 'x':
      def bc_func(xs, t, nn):
        x, y, z = xs[0], xs[1], xs[2]
        u_out = nn
        u_out = u_out.at[0].set(
          x * t * final_displacement / length + x * (x - length) * t * nn[0] / length**2
        )
        u_out = u_out.at[1].set(
          x * (x - length) * t * nn[1] / length**2
        )
        u_out = u_out.at[2].set(
          x * (x - length) * t * nn[2] / length**2
        )
        return u_out
    elif direction == 'y':
      def bc_func(xs, t, nn):
        x, y, z = xs[0], xs[1], xs[2]
        u_out = nn
        u_out = u_out.at[0].set(
          y * (y - length) * t * nn[0] / length**2
        )
        u_out = u_out.at[1].set(
          y * t * final_displacement / length + y * (y - length) * t * nn[1] / length**2
        )
        u_out = u_out.at[2].set(
          y * (y - length) * t * nn[2] / length**2
        )
        return u_out
    elif direction == 'z':
      def bc_func(xs, t, nn):
        x, y, z = xs[0], xs[1], xs[2]
        u_out = nn
        u_out = u_out.at[0].set(
          z * (z - length) * t * nn[0] / length**2
        )
        u_out = u_out.at[1].set(
          z * (z - length) * t * nn[1] / length**2
        )
        u_out = u_out.at[2].set(
          z * t * final_displacement / length + z * (z - length) * t * nn[2] / length**2
        )
        return u_out
    else:
      raise ValueError('Direction must be x, y, or z for this BVP.')
  else:
    raise ValueError('Dimensions can be either 2 or 3.')

  return bc_func
