def BiaxialLinearRamp(
  final_displacement_x: float,
  final_displacement_y: float,
  length_x: float,
  length_y: float
):
  def bc_func(xs, t, z):
    x, y = xs[0], xs[1]
    u_out = z
    # u_out = u_out.at[0].set(
    #   x * (x - length_x) * t * z[0] / length_x**2 + \
    #   x * t * final_displacement_x / length_x
    # )
    # u_out = u_out.at[1].set(
    #   y * (y - length_y) * t * z[1] / length_y**2 + \
    #   y * t * final_displacement_y / length_y
    # )
    u_out = u_out.at[0].set(
      (x * (x - length_x) * t / length_x**2 * \
       y * (y - length_y) * t / length_y**2) * z[0] + \
      x * t * final_displacement_x / length_x
    )
    u_out = u_out.at[1].set(
      (x * (x - length_x) * t / length_x**2 * \
       y * (y - length_y) * t / length_y**2) * z[1]
    )
    return u_out

  return bc_func