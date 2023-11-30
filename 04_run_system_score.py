import os
import json
import pandas as pd
import duckdb
from metaflow import FlowSpec, step, retry, Parameter, JSONType
from risk.risk_utilities import build_scorelist, pca_risk
from datetime import datetime
import io
from io import StringIO


class SystemScore(FlowSpec):
    """Calculates individual and population risk from a design matrix. 
    The design matrix is fetched using the Metaflow Client from run.data.feature_df from the last successful run of the 
    CreateFeatures Flow. The resulting risks are stored in the RiskCreation run
    at run.data.risks and can be accessed using the Metaflow Client.

    Risk creation process benefits from parallelization.  Chanage the processing_groups variable
    to take advantage of available CPU.
    """
    ##PARAMETERS
    regenerate_system_scores = Parameter('regenerate_system_scores',
                        type=bool,
                        help='True to generate scores for all system history, rather than new scores only',
                        default = False)
    set_last_update = Parameter('set_last_update',
                        help='Manually set last_update, controls which scores will be calculated and returned',
                        default = None)
    duckdb_path = Parameter("duckdb_path", 
                            help = 'Path to duckdb database file, if not provided defaults to database in local directory called temp.db', 
                            default='temp.db')
    


    # FLOW STEPS
    @step
    def start(self):
        """
        Create design matrix for entire system.  Either update existing with new data,
        or drop and regenerate all.
        """
        
        ## Parameters
        print("Run Parameters")
        print(f"set_last_update: {self.set_last_update}")
        print(f"regenerate_system_scores: {self.regenerate_system_scores}")
        print(f"duckdb_path: {self.duckdb_path}")

        
        if self.regenerate_system_scores:
            self.last_update = '2023-01-01'
        elif self.set_last_update:
            self.last_update = self.set_last_update
        else:
            self.last_update = self._get_last_update('system_scores')
        
        print(f"System Score Last Run:{self.last_update}")
        
        self._drop_table('system_scores')
        self._drop_table('system_features')
        
        #Aggreget features table to create features for system-wide score
        self.system_features = self._get_system_features()
        self._table_insert(self.system_features.reset_index(), 'system_features')

        self.next(self.calculate_ind_score)

    @step
    def calculate_ind_score(self):
        """
        Calculates individual risks in parallel batches.
        Each batch results saved to user_risk.
        """
        
        self.system_score = pca_risk(['system',self.system_features], 'ind')
        
        self.next(self.end)
    

    @step
    def end(self):
        # Turn risk to score and add score "severity"
        score = ((1 - self.system_score.Risk_ind) * 100).round(2)
        score.name = 'score'
        severity = pd.cut(self.system_score.Risk_ind, bins=[-1,0.66,0.95,0.99,1.0], labels=["low", "medium", "high","very_high"])
        severity.name = 'severity'
        self.system_score = self.system_score.drop(columns = ['Risk_ind'])
        self.system_score = pd.concat([score,severity, self.system_score], axis =1)
        system_score = self.system_score.reset_index()
        with duckdb.connect(self.duckdb_path) as db:
            db.sql(f"CREATE TABLE IF NOT EXISTS system_scores as select * from system_score")
        print('DONE')

    #UTILITY FUNCTIONS
    def _get_system_features(self):
        """Gets column names from feature table,
        then builds and runs the query for system-wide score features
        """
        with duckdb.connect(self.duckdb_path, read_only=True) as db:
            feature_columns = db.sql("select column_name from information_schema.columns where table_name = 'features'").df().column_name.to_list()
        ignore_columns = ['user_id','dt', 'run_ts', 'entity_type']
        query_columns = list(set(feature_columns) - set(ignore_columns))
        query_columns.sort()
        print(query_columns)
        
        with duckdb.connect(self.duckdb_path, read_only=True) as db:
            system_features = db.sql(f"""SELECT 
                                            dt, 
                                            'system' as user_id, 
                                            {self._query_agg_elements(query_columns)} 
                                        FROM features 
                                        GROUP BY dt 
                                        ORDER BY dt""").df()
        return system_features.set_index(['dt', 'user_id'])   
        
    def _query_agg_elements(self, lst:list) -> str:
        """A list of columns is converted to agg statements formatted
        to be inserted into the sql query for system-wide score features

        Args:
            lst (list): list of columns to be aggregated for system score

        Returns:
            str: string inserted into sql query with agg statements for all columns in feature table used in score.
        """
        agg_statements = ''
        for i, el in enumerate(lst):
            if i == 0:
                agg_statements += f'sum({el}) {el}'
            elif i == len(lst):
                agg_statements += f' sum({el}) {el}'
            else:
                agg_statements += f', sum({el}) {el}'
        return agg_statements
    
    def _get_last_update(self, table_name):
        
        with duckdb.connect(self.duckdb_path, read_only=True) as db:      
            try:
                last_update = db.sql(f"select max(run_ts) from {table_name}").fetchall()[0][0]
            except:
                last_update = datetime.fromtimestamp(0) 
            if last_update is None:
                last_update = datetime.fromtimestamp(0) 
            return last_update
        
    def _get_earliest_new_feature_date(self, table_name, last_update):
        with duckdb.connect(self.duckdb_path, read_only=True) as db:      
            try:
                rescore_start_date = db.sql(f"select min(dt) from {table_name} where {last_update} <= run_ts").fetchall()[0][0]
            except:
                rescore_start_date = datetime.fromtimestamp(0) 
            if rescore_start_date is None:
                rescore_start_date = datetime.fromtimestamp(0) 
            return rescore_start_date
    
    def _drop_table(self, table_name):
        with duckdb.connect(self.duckdb_path) as db:
            db.sql(f"DROP TABLE IF EXISTS {table_name}; DROP SEQUENCE IF EXISTS seq_{table_name}")
    
    def _table_insert(self, feature_df, table_name):
        #Create new table of parsed logs to table in database if it doesn't exist
        if len(feature_df) == 0:
            has_new_features = False
        else:
            #feature_df = feature_df.reset_index()
            has_new_features = True
        try:
            #Insert new parsed logs if table does exist
            if has_new_features:
                with duckdb.connect(self.duckdb_path) as db:
                    db.execute(f"INSERT INTO {table_name} SELECT  * FROM feature_df")
                    print(f"Table - {table_name} - updated")
            else:
                print("No update - No new logs to process")
        
        except Exception as e:
            #if catalog does not exist
            if isinstance(e, (duckdb.CatalogException,)):
                with duckdb.connect(self.duckdb_path) as db:
                    db.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM feature_df")
            else:
                raise

if __name__ == '__main__':
    SystemScore()
