import os
import cx_Oracle
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# DB_USER = "JGI8182"
# DB_PASS = "JGI8182"
# # HOST = "192.168.200.174"
# # HOST = "10.255.0.103"
# HOST = "192.168.200.125"

# PORT = "1521"
# SERVICE = "PRASANNA"
# # SERVICE = "ORCL"
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

# query="""
# SELECT SUBSTR(BS_DATE(SALES_DATE),1,7) AS BS_YEAR_MONTH,
# SUM(NVL(QUANTITY*NET_GROSS_RATE, 0)) AS SALES_VALUE
# FROM SA_SALES_INVOICE
# WHERE DELETED_FLAG="N"
# AND COMPANY_CODE IN ('06','0')
# GROUP BY SUBSTR(BS_DATE(SALES_DATE),1,7)
# ORDER BY 1
# """

# #Query for without return items
# query = """
#  SELECT 
#         SUBSTR(BS_DATE(SALES_DATE), 1, 7) AS BS_YEAR_MONTH, 
#         SUM(NVL(QUANTITY, 0)) AS SALES_QTY,
#         SUM(NVL(QUANTITY * NET_GROSS_RATE, 0)) AS SALES_VALUE
#   FROM SA_SALES_INVOICE
#     WHERE DELETED_FLAG = 'N' 
#     AND COMPANY_CODE IN ('01', '0')   
#     GROUP BY SUBSTR(BS_DATE(SALES_DATE), 1, 7)
# ORDER BY BS_YEAR_MONTH
# """

# general query to fetch directly from main table i.e. sa_sales_invoice
# query = """
# SELECT 
#     BS_YEAR_MONTH,
#     SUM(NVL(SALES_QTY, 0)) - SUM(NVL(SALES_RET_QTY, 0)) AS qty,
#     SUM(NVL(SALES_VALUE, 0)) - SUM(NVL(SALES_RET_VALUE, 0)) AS sales
# FROM (
#     -- Sales Data
#     SELECT 
#         SUBSTR(BS_DATE(SALES_DATE), 1, 7) AS BS_YEAR_MONTH, 
#         SUM(NVL(QUANTITY, 0)) AS SALES_QTY,
#         SUM(NVL(QUANTITY * NET_GROSS_RATE, 0)) AS SALES_VALUE,
#         0 AS SALES_RET_QTY,
#         0 AS SALES_RET_VALUE
#     FROM SA_SALES_INVOICE
#     WHERE DELETED_FLAG = 'N' 
#     AND COMPANY_CODE IN ('01', '0')
#     GROUP BY SUBSTR(BS_DATE(SALES_DATE), 1, 7)
#     UNION ALL
#     -- Sales Return Data
#     SELECT 
#         SUBSTR(BS_DATE(RETURN_DATE), 1, 7) AS BS_YEAR_MONTH,
#         0 AS SALES_QTY,
#         0 AS SALES_VALUE,
#         SUM(NVL(QUANTITY, 0)) AS SALES_RET_QTY,
#         SUM(NVL(QUANTITY * NET_GROSS_RATE, 0)) AS SALES_RET_VALUE
#     FROM SA_SALES_RETURN
#     WHERE DELETED_FLAG = 'N' 
#     AND COMPANY_CODE IN ('01', '0')
#     GROUP BY SUBSTR(BS_DATE(RETURN_DATE), 1, 7)
# )
# GROUP BY BS_YEAR_MONTH
# ORDER BY BS_YEAR_MONTH
# """

# # query = """
# SELECT SUBSTR(NEP_SALES_DATE, 1, 7) AS NEPALI_SALES_MONTH, SUM(ITEM_WISE_TOTAL_PRICE) AS TOTAL_PRICE 
# FROM ai_TEST_1 
# WHERE COMPANY_CODE='01'
# GROUP BY STARTDATE, ENDDATE, MONTH, SUBSTR(NEP_SALES_DATE, 1, 7) 
# ORDER BY STARTDATE


# # """

# # Fetching month wise english date
# query = """
# select rangename,startdate,enddate,sum(ITEM_WISE_TOTAL_PRICE), sum(ITEM_WISE_QTY)  from 
# (
# select a.sales_date,a.ITEM_WISE_TOTAL_PRICE,a. ITEM_WISE_QTY, b.startdate,b.enddate,b.rangename from 
# (
# select sales_date,ITEM_WISE_TOTAL_PRICE, ITEM_WISE_QTY from   ai_TEST_1 
# )a
# left outer join
# (
# select startdate ,enddate,rangename from v_date_range_eng
# )b
# on a.sales_date between startdate and enddate
# )
# group by rangename,startdate,enddate
# order by 2
# """

df = pd.read_sql(query, con=connection)
df.to_csv("company_06/hello.csv", index=False)

connection.close()


print(f"Data saved to output_{DB_USER}.csv")

# import pandas as pd

# df = pd.read_csv("output_JGI8182.csv")

# extracted_column = df[['NEPALI_SALES_MONTH','TOTAL_QTY','TOTAL_PRICE']]

# # print(extracted_column)

# extracted_column.to_csv("new_5.csv")

