from metaflow import FlowSpec, Flow, step, retry, Parameter, JSONType
import os
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from importlib import import_module
import primitives

class UpdatePrimitives(FlowSpec):
    #Runtime Parameters
    duckdb_path = Parameter("duckdb_path", 
                            help = 'Path to duckdb database file, if not provided defaults to database in local directory called temp.db', 
                            default='temp.db')
    drop_table = Parameter("drop_table", type=bool, default=False)
    
    ##### FLOW STEPS #####
    @step
    def start(self):
        """
        Begin flow and kick off parallel steps for generating the
        primitives table from parsed log tables.

        """  
        print(f"Database path: {self.duckdb_path}")
        
        #get module names
        self.primitive_modules = primitives.__all__
        self.primitive_table = 'primitives'
        
        #drop tables
        if self.drop_table:
            self._drop_table(self.primitive_table)
        
        print(f'Launching primitive modules: {self.primitive_modules}')
        self.next(self.apply_modules, foreach='primitive_modules')

    @step
    def apply_modules(self):
        """Processes module. Modules run in parallel.
        """
        module_name = self.input
        print(f'Loading module: {module_name}')
        module = import_module(f'primitives.{module_name}')
        primitive_type = module.PRIMITIVE_TYPE
        last_update = self._get_last_update(self.primitive_table, primitive_type)
        
        print(f'Updating primitive Type: {primitive_type}')
        print(f'Last Update for {primitive_type}: {last_update}')
        self.module_output = module.run(self.duckdb_path, last_update)

        self.next(self.collect_modules)
    @step
    def collect_modules(self, inputs):
        primitive_tables = [i.module_output for i in inputs]
        self.merge_artifacts(inputs, include=["duckdb_path"])
        
        #get common fields
        primitive_columns = self._collect_columns(primitive_tables)

        #Set all tables to common fields
        primitive_tables = self._set_columns(primitive_tables, primitive_columns)           

        combined_primitives = pd.concat([table[primitive_columns] for table in primitive_tables]) 
        print(combined_primitives.shape)
        
    
        #Insert Table
        self._table_insert(combined_primitives, 'primitives')
   
        self.next(self.end)

    @step
    def end(self):
        print('DONE')
   
    
    
        
    ##### Utility Functions #####
    def _get_last_update(self, table_name, primitive_type):
        with duckdb.connect(self.duckdb_path, read_only=True) as db:      
            try:
                last_update = db.sql(f"select max(run_ts) from {table_name} where primitive_type = '{primitive_type}'").fetchall()[0][0]
            except:
                last_update = datetime.fromtimestamp(0) 
            if last_update is None:
                last_update = datetime.fromtimestamp(0) 
            return last_update
    
    def _drop_table(self, table_name):
        with duckdb.connect(self.duckdb_path) as db:
            db.sql(f"DROP TABLE IF EXISTS {table_name}; DROP SEQUENCE IF EXISTS seq_{table_name}")
    
    def _table_insert(self, df, table_name):
        #Check for new features
        if len(df) == 0:
            has_new_features = False
        else:
            has_new_features = True
            
        #Check if table exists
        try:
            with duckdb.connect(self.duckdb_path) as db:
                table_exists = db.sql(f"select count(*) from information_schema.tables where table_name = '{table_name}'").df().values[0][0]
        except Exception as e:
            print(e)
            table_exists = 0

        #Get existing column order and add new colums 
        if table_exists: 
            with duckdb.connect(self.duckdb_path) as db:
                    ordered_cols = db.sql(f"select column_name from information_schema.columns where table_name = '{table_name}' order by ordinal_position").df().values[:,0]
                    odered_cols_str = ', '.join(ordered_cols)
            new_cols = set(df.columns) - set(ordered_cols)
            if len(new_cols) > 0:
                print(f"New columns detected: {new_cols}")
                with duckdb.connect(self.duckdb_path) as db:
                    for col in new_cols:
                        db.execute(f"alter table {table_name} ADD COLUMN {col}")
                #reset column list
                with duckdb.connect(self.duckdb_path) as db:
                    ordered_cols = db.sql(f"select column_name from information_schema.columns where table_name = '{table_name}' order by ordinal_position").df().values[:,0]
                    odered_cols_str = ', '.join(ordered_cols)
            
        #Insert data
        try:
            if table_exists & has_new_features:
                with duckdb.connect(self.duckdb_path) as db:
                    db.execute(f"INSERT INTO {table_name} SELECT {odered_cols_str} FROM df")
                    print(f"Table - {table_name} - updated")
                    
            elif table_exists & (has_new_features == False):
                print("No update - No new logs to process")
            else: #table does not exist
                with duckdb.connect(self.duckdb_path) as db:
                    db.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
                    print(f"Table - {table_name} - created")
                
        except Exception as e:
            print(e)
            raise
            
    def _collect_columns(self, primitive_tables):
        """Receives a list of dataframes. Returns a 
        list of all unique column names.

        Args:
            primitive_tables (list[pandas.DataFame]): List of dataframes

        Returns:
            list: List of distinct column names from all tables.
        """
        primitive_columns = []
        for table in primitive_tables:
            for col in table.columns:
                if col not in primitive_columns:
                    primitive_columns.append(col)
        return primitive_columns
    
    def _set_columns(self, primitive_tables, primitive_columns):
        """Recieves list of dataframes and list of columns that need to be shared
        across all dataframes so they can be concatenated.  Columns are added if they do not exist.
        Args:
            primitive_tables (list[pandas.DataFrame]): List of primitive tables as dataframes.
            primitive_columns (list): List of column names
        Returns:
            list[pandas.DataFame]: List dataframes with the same columns in the same order.
        """
        for col in primitive_columns:
            for i, table in enumerate(primitive_tables):
                if col not in table.columns:
                    primitive_tables[i][f'{col}'] = None
        
        #Reorganize table columns to common column order and return 
        return [table[primitive_columns] for table in primitive_tables] 

           
if __name__ == '__main__':
    UpdatePrimitives()