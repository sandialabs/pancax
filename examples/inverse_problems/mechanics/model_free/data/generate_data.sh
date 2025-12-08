module load sierra/daily
sierra adagio -i adagio.inp
module unload sierra/daily

# source ../../../../../venv/bin/activate
python ../../../../../scripts/extract_global_mechanics_data.py \
    --csv-file global_data.csv \
    --displacement-variable displ_y \
    --exodus-file adagio.e \
    --force-variable internal_force_y \
    --nodeset nodelist_5
python ../../../../../scripts/extract_full_field_mechanics_data.py \
    --csv-file full_field_data.csv \
    --exodus-file adagio.e \
    --nodal-variables displ_x,displ_y,displ_z \
    --nodeset nodelist_1