from metaflow import FlowSpec, Flow, step, retry, Parameter, JSONType
import os
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from importlib import import_module
import features


## Changes ##

class CreateFeatures(FlowSpec):
    #Runtime Parameters:::
    duckdb_path = Parameter("duckdb_path", 
                            help = 'Path to duckdb database file, if not provided defaults to database in local directory called temp.db', 
                            default='temp.db')
    drop_table = Parameter("drop_table", type=bool, default=False)
    
    recalculate_after = Parameter("recalculate_after", help = 'Date string in form <yyyy-mm-dd> to recalculate features after', default=None)
    
    ##### FLOW STEPS #####
    @step
    def start(self):
        """
        Begin flow and kick off parallel steps for generating
        features from selected feature modules.

        """ 
        print(f"Database path: {self.duckdb_path}")
        
        #Initialize list for feature module results
        self.feature_dfs = []
        
        feature_table_name = 'features'
        
        #get module names
        self.feature_modules = features.__all__
        
        #drop tables
        if self.drop_table:
            print("Dropping Feature Table")
            self._drop_table(feature_table_name)
        
        if not self.recalculate_after:
            self.last_update = self._get_last_update(feature_table_name)
            print(f"Last update: {self.last_update}")
        else:
            self.last_update = self.recalculate_after
            print(f"Recalculation date set for all features after: {self.last_update}")
        
        self.calculation_date, self.oldest_record = self._get_calculation_date(self.last_update, look_back_days = 7)
        print(f"Oldest unprocessed primitives: {self.oldest_record}")
        print(f"Recalculating features from primitives occuring on or after: {self.calculation_date}")
        print(f'Launching feature modules: {self.feature_modules}') #todo
        self.next(self.apply_modules,foreach='feature_modules')
    

    @step
    def apply_modules(self):
        """
        Generate features from feature module
        """
        module_name = self.input
        print(f'Loading module: {module_name}')
        module = import_module(f'features.{module_name}')
 
        self.module_output = module.run(self.duckdb_path, self.calculation_date)

        self.next(self.collect_modules)


    @step
    def collect_modules(self, inputs):
        """
        Join features into a common dataframe .

        """
        feature_dfs = [i.module_output for i in inputs]
        self.merge_artifacts(inputs, include=['duckdb_path', 'calculation_date', 'oldest_record'])
        print("Joining features to common dataframe")
        print(f"Shapes before joining: {[df.shape for df in feature_dfs]}")
        self.feature_df = pd.concat(feature_dfs, axis=1).fillna(0).reset_index()
        print(f"Shape after joining:  \n{self.feature_df.shape}")
        
        #drop feature days included for trailing aggregations
        self.feature_df = self.feature_df[self.feature_df['dt'] >= self.oldest_record]
        print(f"Shape after dropping look_back_days:  \n{self.feature_df.shape}")
        
        self.feature_df = self._filter_out_today(self.feature_df)
        print("Filter out current date activity")
        print(self.feature_df.shape)
        self.next(self.end)

    @step
    def end(self):
        #Delete records that were recalculated
        self._crop_table(self.oldest_record, 'features')
        #Insert Records
        self._table_insert(self.feature_df,'features')
        print('DONE')
        
    ##### Utility Functions #####
    def _get_last_update(self, table_name):
        with duckdb.connect(self.duckdb_path, read_only = True) as db:    
            try:
                last_update = db.sql(f"select max(dt) from {table_name}").fetchall()[0][0]
            except:
                last_update = '2023-03-01'#datetime.fromtimestamp(0)
            return last_update
        
    def _get_calculation_date(self, recalculate_after, look_back_days = 7):
        """Calculate all features after given date parameter.  In order to accomplish this
        we must find the oldest primitive_ts that was captured after the given date.
        We then must account for any trailing aggregations - so calculation date will be
        The oldest primitive_ts minus the required look_back_days to correctly calculate any
        trailing aggregations.
        """
        DEFAULT_START = '2023-03-01'
        if not recalculate_after:
            with duckdb.connect(self.duckdb_path, read_only =True) as db:
                try: 
                    oldest_record = db.sql(f""" 
                        WITH last_run as(
                        SELECT 
                            max(run_ts) last_run
                        FROM features)
                        
                        SELECT
                            min(new_record_dates) oldest_record
                        FROM (Select 
                                cast(primitive_ts as date) new_record_dates,
                                run_ts
                            from primitives) a
                        join last_run b
                        on a.run_ts > b.last_run
                        """).fetchall()[0][0]
                    
                    calculation_date = oldest_record - timedelta(days = look_back_days)
                    return str(calculation_date), str(oldest_record)
                except Exception as e:
                    print(e)
                    return '2023-03-01', '2023-03-01'
        else:
            oldest_record = datetime.strptime(recalculate_after, '%Y-%m-%d').date()
            calculation_date = oldest_record - timedelta(days = look_back_days)
            return str(calculation_date), str(oldest_record)
            
    
    def _drop_table(self, table_name):
        with duckdb.connect(self.duckdb_path) as db:
            db.sql(f"DROP TABLE IF EXISTS {table_name};")
    
    def _crop_table(self, calculation_date, table_name):
        try:
            with duckdb.connect(self.duckdb_path) as db:
                table_exists = db.sql(f"select count(*) from information_schema.tables where table_name = '{table_name}'").df().values[0][0]
        except:
            table_exists = 0   
        if table_exists:    
            with duckdb.connect(self.duckdb_path) as db:
                rows_before = db.sql(f"select count(*) n from {table_name}").fetchall()[0][0]
                db.execute(f"DELETE FROM {table_name} WHERE dt >= '{calculation_date}'")
                rows_after = db.sql(f"select count(*) n from {table_name}").fetchall()[0][0]
                print(f"Deleteing All Rows After: {calculation_date}")
                print(f"Row count before crop: {rows_before}")
                print(f"Row count after crop: {rows_after}")
        else:
            print("No table to crop")
    
    def _table_insert(self, feature_df, table_name):
        
        #Create new table of parsed logs to table in database if it doesn't exist
        if len(feature_df) == 0:
            has_new_features = False
        else:
            feature_df = feature_df
            has_new_features = True
            
        try:
            with duckdb.connect(self.duckdb_path) as db:
                table_exists = db.sql(f"select count(*) from information_schema.tables where table_name = '{table_name}'").df().values[0][0]
        except:
            table_exists = 0       
            
        
        if not table_exists and has_new_features:
            db = duckdb.connect(self.duckdb_path)
            db.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM feature_df")
            db.close()
            print(f"New Table - {table_name} created.")

        #Insert new parsed logs if table does exist
        elif table_exists and has_new_features:
            with duckdb.connect(self.duckdb_path) as db:
                rows_before = db.sql(f"select count(*) n from {table_name}").fetchall()[0][0]
                db.execute(f"INSERT INTO {table_name} SELECT  * FROM feature_df")
                rows_after = db.sql(f"select count(*) n from {table_name}").fetchall()[0][0] 
                print(f"Table - {table_name} - updated")
                print(f"Row count before insert: {rows_before}")
                print(f"Row count after insert: {rows_after}")
        else:
            print("No update - No new logs to process")
            
    
    def _filter_out_today(self, df): 
        return df[df['dt'] != str(datetime.today().date())]

           
if __name__ == '__main__':
    CreateFeatures()

