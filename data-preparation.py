from utils import authenticate
from sklearn.model_selection import train_test_split
import pandas as pd
import vertexai
import datetime

credentials, PROJECT_ID = authenticate() 
REGION = "us-central1"
vertexai.init(project = PROJECT_ID,
              location = REGION,
              credentials = credentials)

from google.cloud import bigquery
#First download and copy this file from your GCP Project->Service Accounts->Keys in GCP console.
bq_client = bigquery.Client.from_service_account_json("gcp-project-service-account.json") 

#Stack Overflow Public Dataset

QUERY_TABLES = """
SELECT
  table_name
FROM
  `bigquery-public-data.stackoverflow.INFORMATION_SCHEMA.TABLES`
"""
query_job = bq_client.query(QUERY_TABLES)
for row in query_job:
    for value in row.values():
        print(value)

#Data Retrieval
        
INSPECT_QUERY = """
SELECT
    *
FROM
    `bigquery-public-data.stackoverflow.posts_questions`
LIMIT 3
"""
query_job = bq_client.query(INSPECT_QUERY)

stack_overflow_df = query_job\
    .result()\
    .to_arrow()\
    .to_pandas()
#stack_overflow_df.head()
print(stack_overflow_df)

#Dealing with Large Datasets

QUERY_ALL = """
SELECT
    *
FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
"""

query_job = bq_client.query(QUERY_ALL)
try:
    stack_overflow_df = query_job\
    .result()\
    .to_arrow()\
    .to_pandas()
except Exception as e:
    print('The DataFrame is too large to load into memory.', e)
#Note: The data is too large to return, as it is not fitting into memory.
#Joining Tables and Query Optimization
    
QUERY = """
SELECT
    CONCAT(q.title, q.body) as input_text,
    a.body AS output_text
FROM
    `bigquery-public-data.stackoverflow.posts_questions` q
JOIN
    `bigquery-public-data.stackoverflow.posts_answers` a
ON
    q.accepted_answer_id = a.id
WHERE
    q.accepted_answer_id IS NOT NULL AND
    REGEXP_CONTAINS(q.tags, "python") AND
    a.creation_date >= "2020-01-01"
LIMIT
    10000
"""
query_job = bq_client.query(QUERY)
### this may take some seconds to run
stack_overflow_df = query_job.result()\
                        .to_arrow()\
                        .to_pandas()

print(stack_overflow_df.head(2))

#Adding Instructions
INSTRUCTION_TEMPLATE = f"""\
Please answer the following Stackoverflow question on Python. \
Answer it like you are a developer answering Stackoverflow questions.

Stackoverflow question:
"""

stack_overflow_df['input_text_instruct'] = INSTRUCTION_TEMPLATE + ' '\
    + stack_overflow_df['input_text']

#Dataset for Tuning
train, evaluation = train_test_split(
    stack_overflow_df,
    ### test_size=0.2 means 20% for evaluation
    ### which then makes train set to be of 80%
    test_size=0.2,
    random_state=42
)

#Different Datasets and Flow
date = datetime.datetime.now().strftime("%H:%d:%m:%Y")
#Generate a jsonl file.
cols = ['input_text_instruct','output_text']
tune_jsonl = train[cols].to_json(orient="records", lines=True)

training_data_filename = f"tune_data_stack_overflow_\
                            python_qa-{date}.jsonl"

training_data_filename = training_data_filename.replace(":","_")
with open(training_data_filename, "w") as f:
    f.write(tune_jsonl)

