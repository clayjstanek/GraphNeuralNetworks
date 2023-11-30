source .env/bin/activate

python 01_run_parsers.py run --duckdb_path $DUCKDB_DEV_PATH --drop_all_tables True
python 02_run_primitives.py run --duckdb_path $DUCKDB_DEV_PATH --drop_table True
python 03_run_features.py run --duckdb_path $DUCKDB_DEV_PATH --drop_table True
python 04_run_system_score.py run --duckdb_path $DUCKDB_DEV_PATH