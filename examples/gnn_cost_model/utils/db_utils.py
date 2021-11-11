import sqlite3
from sqlite3 import Error


def get_database_size(cursor, table):
    return cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]


def get_all_states(cursor, db_size):
    if db_size == -1:
        cursor.execute("SELECT * from States")
    else:
        cursor.execute(f"SELECT * from States LIMIT {db_size}")

    return cursor.fetchall()


def get_observation_from_table(cursor, hash):
    """
    Gets the observation for a state_id from a given database
    Inputs:
        - cursor: the db cursor
        - state_id: the state_id we want (primary key in the table)
    """
    cursor.execute(f"SELECT * from Observations where state_id = '{hash}'")
    return cursor.fetchall()
