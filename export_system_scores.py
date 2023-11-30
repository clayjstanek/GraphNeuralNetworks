"""
This script queries system scores from duckdb, converts to json files and writes to destination.

Args:
    --dest_path (str): Destination directory for json file
    --duckdb_path (str): Path of duckdb database.
    
Usage:
    python export_system_scores.py --dest_path '/telemetry/scores' --duckdb_path '/duckdb/mydb.db'
"""

import os
import json
import datetime
from pathlib import Path
import pandas as pd
import duckdb
from utilities import query


def scores_to_json(df: pd.DataFrame) -> dict:
    """Converts score summary dataframe to json records.

    Args:
        df (pd.DataFrame): Results from query of duckdb.system_scores table.

    Returns:
        dict: dicionary of records.  where the key is the filename and the value is a the record in the form of a python dictionary
        that can be serialized json.
    """
    now = datetime.datetime.now()
    now_str = now.strftime(("%Y-%m-%d_%H_%M_%S"))

    records = {}
    for idx, row in df.iterrows():
        if True:
            user = idx[0]      
            timestamp = pd.to_datetime(idx[1]).isoformat()
            score_date = idx[1].date()
            company = "Peraton"
            persona = "system"
            if pd.isna(row.score):
                score = None
            else:
                score = row.score
            severity = row.severity
            name1 = row.Var1_ind   
            name2 = row.Var2_ind        
            name3 = row.Var3_ind        
            value1 = row.Var1Val_ind     
            value2 = row.Var2Val_ind     
            value3 = row.Var3Val_ind    
            mean1 = row.Var1Mean_ind    
            mean2 = row.Var2Mean_ind    
            mean3 = row.Var3Mean_ind 
            #last_login_days = val.last_login_days
            if severity == 'low':
                message = 'No Issues'
            elif score == None:
                severity = None
                message = {
                    "top_3_reasons": {
                            "reason_1": {
                                "name":None,
                                "value":None,  
                                "mean":None
                            }, 
                            "reason_2": {
                                "name":None,
                                "value":None,  
                                "mean":None
                            }, 
                            "reason_3": {
                                "name":None,
                                "value":None,  
                                "mean":None
                            }
                        }
                    }
            else:
                message = {
                    "top_3_reasons": {
                            "reason_1": {
                                "name":name1,
                                "value": value1,  
                                "mean":mean1
                            }, 
                            "reason_2": {
                                "name":name2,
                                "value": value2,  
                                "mean":mean2
                            }, 
                            "reason_3": {
                                "name":name3,
                                "value": value3,  
                                "mean":mean3
                            }
                        }
                    }

            record = {
                "timestamp": timestamp,
                "company": company,
                "user": user,
                "persona": persona,
                "score": score,
                #"last_login": last_login_days,
                "event_data": {
                    "severity": severity,
                    "message": message            
                    }
                }
        
            records[f'{user}/{now.date()}/{company}_{persona}_{user}_{score_date}_{now_str}.json'] = record
        else:
            pass
    return records


def write_records(records: dict, base_path: str) -> None:
    """Converts records to json and writes to path.  Expects records object
    that has the filename as the key, and json seriralizable data as the value.
    Automatically creates new directories if they do not exist.

    Args:
        records (dict): Dictionary where key is file path and value is a json serializable record. 
        base_path (str): path where scores are kept
    """
    for k, v in records.items():
        # Create new directories if needed
        #path_lst = f'{base_path}'.split('/')
        new_path = f'{k}'.split('/')[:-1]
        #path_lst.extend(new_path)
        path_str = os.path.join(base_path,*new_path)
        print(path_str)
        Path(path_str).mkdir(parents=True, exist_ok=True)
        # Create json 
        json_outfile = json.dumps(v)
        #write to path
        with open(f'{base_path}/{k}','w') as outfile:
            outfile.write(json_outfile)


if __name__ == '__main__':
    import argparse
    
    #Add parsers for runtime arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest_path', help= 'Destination directory for json score files.')
    parser.add_argument('--duckdb_path', help='Path to duckdb database file')
    
    #Parse Runtime Arguments
    args = parser.parse_args()
    dest_path = args.dest_path
    duckdb_path = args.duckdb_path
    
    # query data from duckdb
    df = query("select * from system_scores", path=duckdb_path)
    
    #transform to json
    records = scores_to_json(df.set_index(['user_id', 'dt']))
    
    #write to file
    write_records(records, dest_path)