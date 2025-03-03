import os
import cx_Oracle
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
# loading credentials
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
SERVICE = os.getenv('SERVICE')

print(f"Values are:{DB_USER}\n{DB_PASS}\n{HOST}\n{PORT}\n{SERVICE}")


dsn_tns = cx_Oracle.makedsn(HOST, PORT, service_name=SERVICE)
connection = cx_Oracle.connect(user=DB_USER, password=DB_PASS, dsn=dsn_tns)


query="""
 SELECT 
        SUBSTR(BS_DATE(SALES_DATE), 1, 7) AS BS_YEAR_MONTH, 
        SUM(NVL(QUANTITY * NET_GROSS_RATE, 0)) AS SALES_VALUE
  FROM SA_SALES_INVOICE
    WHERE DELETED_FLAG = 'N' 
    AND COMPANY_CODE IN ('06', '0')   
    GROUP BY SUBSTR(BS_DATE(SALES_DATE), 1, 7)
ORDER BY 1
"""
df = pd.read_sql(query, con=connection)
df.to_csv("company_06/hello.csv", index=False)

connection.close()
print(f"Data saved to output_{DB_USER}.csv")