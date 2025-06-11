# from pancax import GlobalData
# from pathlib import Path
# import os
import pytest


def test_global_data():
    from pancax import GlobalData
    from pathlib import Path
    import os
    data_file = os.path.join(Path(__file__).parent, 'data_global.csv')
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    data = GlobalData(
        data_file, 
        times_key='t',
        disp_key='u_x',
        force_key='f_x',
        mesh_file=mesh_file,
        nset_id=6,
        reaction_dof='x',
        n_time_steps=11
    )
    data = GlobalData(
        data_file, 
        times_key='t',
        disp_key='u_x',
        force_key='f_x',
        mesh_file=mesh_file,
        nset_id=6,
        reaction_dof='y',
        n_time_steps=11
    )
    data = GlobalData(
        data_file, 
        times_key='t',
        disp_key='u_x',
        force_key='f_x',
        mesh_file=mesh_file,
        nset_id=6,
        reaction_dof='z',
        n_time_steps=11
    )


@pytest.mark.skip(reason='Failing on missions with bad tk')
def test_global_data_with_plotting():
    from pancax import GlobalData
    from pathlib import Path
    import os
    data_file = os.path.join(Path(__file__).parent, 'data_global.csv')
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    data = GlobalData(
        data_file, 
        times_key='t',
        disp_key='u_x',
        force_key='f_x',
        mesh_file=mesh_file,
        nset_id=6,
        reaction_dof='x',
        n_time_steps=11,
        plotting=True
    )


def test_global_data_bad_reaction_dof():
    from pancax import GlobalData
    from pathlib import Path
    import os
    data_file = os.path.join(Path(__file__).parent, 'data_global.csv')
    mesh_file = os.path.join(Path(__file__).parent, 'mesh.g')
    with pytest.raises(ValueError):
        data = GlobalData(
            data_file, 
            times_key='t',
            disp_key='u_x',
            force_key='f_x',
            mesh_file=mesh_file,
            nset_id=6,
            reaction_dof=0,
            n_time_steps=11
        )
