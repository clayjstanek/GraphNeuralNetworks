import os
import json
import pandas as pd
import duckdb
from metaflow import FlowSpec, step, retry, Parameter, JSONType
from risk.risk_utilities import build_scorelist, pca_risk
import datetime as dt
import io
from io import StringIO

class RiskCreation(FlowSpec):
    """Calculates individual and population risk from a design matrix. 
    The design matrix is fetched using the Metaflow Client from run.data.feature_df from the last successful run of the 
    CreateFeatures Flow. The resulting risks are stored in the RiskCreation run
    at run.data.risks and can be accessed using the Metaflow Client.

    Risk creation process benefits from parallelization.  Chanage the processing_groups variable
    to take advantage of available CPU.
    """
    ##PARAMETERS
    run_all = Parameter('replace_feature_library',
                         help='True to generate risk for all user history, rather than new risk scores only',
                         default = False)

    test = Parameter('test',
                         help='Does not load results to s3 if True',
                         default = False)
    
    set_last_update = Parameter('set_last_update',
                         help='Manually set last_update, controls which scores will be calculated and returned',
                         default = None)


    # FLOW STEPS
    @step
    def start(self):
        """
        Load design matrix and group data by user and date for
        use in individual and population risk calculations.

        """
        from metaflow import Flow
        
        ## Parameters
        processing_groups = 1 #os.cpu_count() -1
        self.bucket = 'hd-datalake'
        self.subpath = 'out/individual-risk'
        self.testpath = './RiskOutput'
        print("Run Parameters")
        print(f"test: {self.test}")
        print(f"run_all: {self.run_all}")
        print(f"set_last_update: {self.set_last_update}")
        print(f"bucket: {self.bucket}")
        print(f"subpath: {self.subpath}")
        
        #Fetch desi
        run = Flow('CreateFeatures').latest_successful_run
        print("Using feature_df from '%s'" % str(run))
        self.feature_df = run.data.feature_df
        with open(f'./DesignMatrix/DM_{run.id}.html','w') as f:
            f.write("""<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css">""")
            html_tbl = self.feature_df.to_html(classes='table table-striped')
            f.write(html_tbl)
            f.write("""<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/js/bootstrap.min.js"></script>""")
        print(f"Feature DF shape: {self.feature_df.shape}")
        #print(f"{self.feature_df.head().to_string()}")
        
        if self.run_all:
            self.last_update = '2022-01-01'
        elif self.set_last_update:
            self.last_update = self.set_last_update
        else:
            self.last_update = '2022-01-01' #self._get_last_update(self.bucket, self.subpath)
        
        print(f"Calculating Risk Scores After:{self.last_update}")
        #Group by user for individual risk
        
        grouped_by_user = list(self.feature_df.groupby(level=0))
        self.batched_users = self._batch_it(grouped_by_user, processing_groups)

        # Group by date for population risk
        new_features = self.feature_df.loc[self.feature_df.index.get_level_values(level=1) > self.last_update]
        print(new_features.shape)
        grouped_by_date = list(new_features.groupby(level=1))
        self.batched_dates = self._batch_it(grouped_by_date, processing_groups)

        self.next(self.calculate_ind_risks, foreach='batched_users')

    @step
    def calculate_ind_risks(self):
        """
        Calculates individual risks in parallel batches.
        Each batch results saved to user_risk.

        """

        user_batch = self.input
        # print(user_group.head())
        batch_risks = []
        for user in user_batch:
            batch_risks.append(pca_risk(user, 'ind'))

        self.user_risk = pd.concat(batch_risks)
        self.next(self.collect_ind_risks)

    @step
    def collect_ind_risks(self, inputs):
        """
        Collect individual risk batches and merge to single dataframe.
        """
        import pandas as pd

        self.user_risks = [i.user_risk for i in inputs]

        print("Joining features to common dataframe")
        print(f"Shapes before joining: {[i.shape for i in self.user_risks]}")
        self.user_risk = pd.concat(self.user_risks)
        print(f"Shape after joining:  \n{self.user_risk.shape}")

        self.merge_artifacts(inputs)

        self.next(self.start_pop_risks)

    @step
    def start_pop_risks(self):
        """Kick off parallel computation of population risk
        """

        self.next(self.calculate_pop_risks, foreach='batched_dates')

    @step
    def calculate_pop_risks(self):
        """
        Calculate population risk in parallel batches.
        Batch data saved to pop_risk.

        """
        # Group by date

        date_batch = self.input
        # print(user_group.head())
        batch_risks = []
        for _date in date_batch:
            batch_risks.append(pca_risk(_date, 'pop'))

        self.pop_risk = pd.concat(batch_risks)
        self.next(self.collect_pop_risks)

    @step
    def collect_pop_risks(self, inputs):
        """
        Merge population risk batches to single dataframe.
        Combine poplulation and individual risk to single dataframe.
        Save final results to self.risks.
        """
        import pandas as pd

        self.pop_risks = [i.pop_risk for i in inputs]

        print("Joining population risks to common dataframe")
        print(f"Shapes before joining: {[i.shape for i in self.pop_risks]}")
        self.pop_risk = pd.concat(self.pop_risks)
        print(f"Shape after joining:  \n{self.pop_risk.shape}")

        self.merge_artifacts(inputs)
        print("Joining population and individual risks to common dataframe")
        print(f"Shapes before joining: {self.pop_risk.shape}, {self.user_risk.shape}")
        self.risks = pd.concat([self.user_risk, self.pop_risk], axis=1)

        print(f"Shape after joining: {self.risks.shape}")

        self.next(self.end)

    @step
    def end(self):
        #Create rows and fill missing active user dates with 0's
        #Creating after risk creation means zero activity days not incorporated into scores.
        user_dates = self._current_active_user_dates()
        
        #Filter out records prior to last_update
        user_dates = user_dates.loc[user_dates.index.get_level_values(level=1) > self.last_update]
        self.user_risk = user_dates.merge(self.user_risk, how='left', left_index =True, right_index=True) 
        #Tag entries with no data, and fill default risk score
        self.user_risk['HasData'] = ~self.user_risk.Risk_ind.isna() 
        #self.user_risk['Risk_ind'] = self.user_risk.Risk_ind.fillna(0)
        self.user_risk = self._get_last_login(self.user_risk)
        #filter out any entries for today(data hasn't been collected so score isn't valid)
        self.user_risk = self._filter_out_today(self.user_risk)
        
        
        #Generate Experience Score JSON Records
        self.json_objects = self.package_risk_ind(self.user_risk)
        print(f"Records: {len(self.json_objects)}")
        #s3 = boto3.resource('s3')
        #s3.Bucket(self.bucket)
        for key, json_object in self.json_objects.items():
            if self.test:
                if not os.path.exists(self.testpath):
                    os.makedirs(self.testpath)
                json_object = json.dumps(json_object)
                # Writing to sample.json
                with open(f"{self.testpath}/{key}", "w") as outfile:
                    outfile.write(json_object)
            
            else:
                pass
                #s3.Object(self.bucket, f'{self.subpath}/{key}').put(Body=json.dumps(json_object))
        if self.test:
            print("TEST: No records posted to S3")
        else:
            print("Records posted to S3")
        print('DONE')

    #UTILITY FUNCTIONS
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
    def _filter_out_today(self, df): 
        return df[df.index.get_level_values('dt') != str(dt.datetime.today().date())]

    def _current_active_user_dates(self):
        """
        Create index with all valid dates for current active users
        Returns:
            pandas multi-index:  Returns a with all valid user * dt combinations
        """ 
           
        user_dates_q = """
        with active_users as (
            select
                profile.login as "user",
                CAST(activated AS DATE) as activated,
                range(0, date_diff('day',cast(activated as DATE),current_date) - 1) days_seq
            from users
            where status = 'ACTIVE'
        ),
        cross_join as(
            select
                "user",
                "day"
            from active_users
            cross join unnest(days_seq) as t(day)
        )
        select a.user, a.activated + b.day dt
        from (select "user", activated from active_users) as a
        join (select "user", cast("day" as INTEGER) "day" from cross_join) as b
        on a.user = b.user
        order by a.user, a.activated + b.day
        """
        db = duckdb.connect('holodeck.db')
        user_dates = db.sql(user_dates_q).df()
        db.close()
        user_dates['dt'] = user_dates['dt'].astype('str')
        user_dates = user_dates.set_index(['user','dt'])
        return(user_dates)
    
    # def _get_last_update(self, bucket, subpath):
    #     #TODO: Make This Solution More Robust (Risks to Athena Table?)
    #     get_score_date = lambda obj: obj['Key'].split('_')[-5]
    #     s3 = boto3.client('s3')
    #     paginator = s3.get_paginator( "list_objects" )
    #     page_iterator = paginator.paginate( Bucket = bucket, Prefix = subpath)
    #     max_lst = []
    #     for page in page_iterator:
    #         if "Contents" in page:
    #             objects = list(filter(lambda obj: obj['Size'] > 0, page["Contents"])) #filtered objects
    #             last_scored_object = sorted(objects, key=get_score_date)[-1]
    #             last_score = get_score_date(last_scored_object)
    #             max_lst.append(last_score)

    #     result = max(max_lst)
    #     return result

    def _get_last_login(self, df):
        df = df.reset_index()
        df['NoData'] = df.HasData.eq(0) 
        df['is_new_user'] = df.user != df.user.shift(1)
        df['newuser_or_hasdata'] = df.is_new_user | df.HasData
        df['cum_groups'] = df.newuser_or_hasdata.cumsum()
        df['last_login_days'] = df.groupby('cum_groups').NoData.transform('cumsum')
        df = df.drop(columns=['NoData', 'is_new_user','newuser_or_hasdata','cum_groups'])
        df = df.set_index(['user','dt'])
        return df  
    
    def package_risk_ind(self, df):
        now = dt.datetime.now()
        now_str = now.strftime(("%Y-%m-%d_%H_%M_%S"))
        df['severity'] = pd.cut(df.Risk_ind, bins=[-1,0.5,0.75,0.95,1.0], labels=["low", "medium", "high","very_high"])
        records = {}
        for idx, val in df.iterrows():
            if True:
                user = idx[0]
                timestamp = pd.to_datetime(idx[1]).isoformat()
                score_date = idx[1]
                company = "Peraton"
                persona = "generalist"
                if pd.isna(val.Risk_ind):
                    score = None
                else:
                    score = round(100 - (val.Risk_ind *100))
                severity = val.severity
                name1 = val.Var1_ind   
                name2 = val.Var2_ind        
                name3 = val.Var3_ind        
                value1 = val.Var1Val_ind     
                value2 = val.Var2Val_ind     
                value3 = val.Var3Val_ind    
                mean1 = val.Var1Mean_ind    
                mean2 = val.Var2Mean_ind    
                mean3 = val.Var3Mean_ind 
                last_login_days = val.last_login_days
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
                    "last_login": last_login_days,
                    "event_data": {
                        "severity": severity,
                        "message": message            
                        }
                    }
            
                records[f'{company}_{persona}_{user}_{score_date}_{now_str}.json'] = record
            else:
                pass
        return records
        
if __name__ == '__main__':
    RiskCreation()
