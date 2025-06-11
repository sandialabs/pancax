def test_full_field_data():
    from pancax import FullFieldData
    from pathlib import Path
    import os
    data_file = os.path.join(Path(__file__).parent, 'data_full_field.csv')
    data = FullFieldData(
        data_file, 
        input_keys=['x', 'y', 'z', 't'],
        output_keys=['u_x', 'u_y', 'u_z']
    )


# def test_full_field_data_plot_registration():
#     data_file = os.path.join(Path(__file__).parent, 'data_full_field.csv')
#     data = FullFieldData(
#         data_file, 
#         input_keys=['x', 'y', 'z', 't'],
#         output_keys=['u_x', 'u_y', 'u_z']
#     )
#     data.plot_e