"""Parse logs to duckdb tables, using paring modules in parsers project directory.

Args:
    --duckdb_path (str): path to duckdb database. Default: './temp.db'
    --drop_tables (JSONtype): JSON string of tables to drop. 
        Example: '["table1", "table2"]'
    --drop_all_tables (bool): True and all tables are dropped and rebuilt from existing logs.
        False and all tables are updated with any unprocessed logs. Default: False.
Usage: To run on your dev database and rebuild all the tables.
    python 01_run_parsers.py run --duckdb_path $DUCKDB_DEV_PATH --drop_all_tables True
"""
#imports
import os
import pandas as pd
import duckdb
from datetime import datetime
from importlib import import_module
from metaflow import FlowSpec, Parameter,step, JSONType

#local imports
import parsers


class ParseData(FlowSpec):
    """
    A flow to parse logs into duckdb tables for creation of primitives
    """
    
    #Runtime Parameters
    duckdb_path = Parameter("duckdb_path", 
                            help = 'Path to duckdb database file, if not provided defaults to database in local directory called temp.db', 
                            default='temp.db')
    drop_tables = Parameter("drop_tables", type=JSONType, default='[]')
    drop_all_tables = Parameter("drop_all_tables", type=bool, default=False)

    ##FLOW STEPS
    @step
    def start(self):
        """
        Begin flow and kick off parallel steps for creating parsed log tables from parser modules.
        """
        #Echo runtime parameters
        print(f"Runtime Parameters: \n\tduckdb_path: {self.duckdb_path}\n\tdrop_tables: {self.drop_tables}") 
        
        #get module names
        self.parser_modules = parsers.__all__
        
        #drop tables
        self._drop_tables(self.parser_modules, self.drop_tables, self.drop_all_tables)
       
        print(f'Launching parsers: {self.parser_modules}')
        self.next(self.apply_parsers, foreach='parser_modules')

    @step
    def apply_parsers(self):
        ## Get Details
        module_name = self.input
        table_name = f'{module_name}_raw'
        
        #Check if table exists and when it was last updated
        last_parse = self._get_last_parse(table_name)
        print(f"last_parse: {last_parse}")
        
        #fetch parsing module and parse new logs
        print(f"Running Parser: {module_name}")
        module = import_module(f'parsers.{module_name}') 
        self.module_output = module.run(self.duckdb_path, table_name, last_parse)
        
        
        
        self.next(self.collect_parsed_logs)
        
    @step
    def collect_parsed_logs(self, inputs):
        module_outputs = [i.module_output for i in inputs]
        self.merge_artifacts(inputs, include=["duckdb_path", "drop_tables","drop_all_tables"])
        #Insert parsed logs to table
        for module_output in module_outputs:
            self._table_insert(module_output)
        print(f"Parsing Complete: {len(module_outputs)} tables processed")
        self.next(self.end)

    
    @step
    def end(self):
        print('DONE')
    
    
    ### Helper functions attached to flow  
    def _drop_table(self, table_name):
        with duckdb.connect(self.duckdb_path) as db:
            db.sql(f"DROP TABLE IF EXISTS {table_name}; DROP SEQUENCE IF EXISTS seq_{table_name}")
    
    def _drop_tables(self, parser_modules, drop_tables, drop_all_tables):
        if len(drop_tables) > 0:
            with duckdb.connect(self.duckdb_path) as db:
                for table_name in drop_tables:
                    print(f"Dropping table: {table_name}")
                    db.sql(f"DROP TABLE IF EXISTS {table_name}; DROP SEQUENCE IF EXISTS seq_{table_name}") 
        if drop_all_tables:
            tables = [f'{parser_module}_raw' for parser_module in parser_modules]
            with duckdb.connect(self.duckdb_path) as db:
                for table_name in tables:
                    print(f"Dropping table: {table_name}")
                    db.sql(f"DROP TABLE IF EXISTS {table_name}; DROP SEQUENCE IF EXISTS seq_{table_name}") 
        
    def _get_last_parse(self, table_name):
        with duckdb.connect(self.duckdb_path, read_only=True) as db:
            try:
                last_parse = db.sql(f"select max(parse_ts) from {table_name}").fetchall()[0][0]
                if not last_parse:
                    last_parse = datetime.fromtimestamp(0)
            except:
                last_parse = datetime.fromtimestamp(0)
            return last_parse
        
        
    def _table_insert(self, module_output):
        #Create new table of parsed logs to table in database if it doesn't exist
        parsed_logs_df, table_seq_query, table_creation_query, table_insert_query, table_name = module_output
        if len(parsed_logs_df) == 0:
            has_new_logs = False
        else:
            has_new_logs = True

        if has_new_logs:
            with duckdb.connect(self.duckdb_path) as db:
                db.execute(table_seq_query)
                db.execute(table_creation_query)
                db.execute(table_insert_query)
     
            print(f"Table {table_name} Updated - {len(parsed_logs_df)} new entries.")
    
        else:
            print("No update - No new logs to process")
            
    def _batch_it(self, lst, batches):
        """Breaks an iterable into a list of batches.
        Args:
            lst (iterable): Iterable such as list, or pandas.DataFrame.Groupby object.
            batches (int): Desired batches to divide iterable.
        Returns:
            list: list of batched iterables.  
        """
        batch_size = int(len(lst)/batches)
        if not batch_size:
            batch_size = 1
        batched = []
        for i in range(0, len(lst), batch_size):
            batched.append(lst[i:i + batch_size])
        return batched

if __name__ == '__main__':
    ParseData()