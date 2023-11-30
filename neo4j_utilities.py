from neo4j import GraphDatabase
import pandas as pd
import json
import logging


class Connection:
    def __init__(self, uri, auth):
        logging.info(f"Connecting to database...")
        self.__uri = uri
        self.__user = auth[0]
        self.__pwd = auth[1]
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
            logging.info(f"Connected to database")
        except Exception as error:
            logging.error(f"Error while connecting to Neo4j")
            logging.error(error)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, parameters=None, dtype = 'list', db=None):
        
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            #logging.info(query)
            if dtype == 'list':
                response = list(session.run(query))
            elif dtype == 'df':
                response = session.run(query).to_df()
            elif dtype == 'json':
                response = json.dumps(session.run(query))
            else:
                response = list(session.run(query))
        except Exception as error:
            logging.error(error)
            logging.error(f"Query failed")
        finally: 
            if session is not None:
                session.close()
        return response