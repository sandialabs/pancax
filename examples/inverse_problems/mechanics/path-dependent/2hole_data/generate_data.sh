module load sierra/daily
sierra adagio -i uniaxial_tension_hyperviscoelasticity.i
module unload sierra/daily

python process_data.py
python plot_force_disp.py \
    --displacement-variable displ_y \
    --force-variable internal_force_y \
    --sierra-exodus-file uniaxial_tension_hyperviscoelasticity.e \
    --sierra-nodeset nodelist_5
