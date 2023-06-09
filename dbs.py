import pymysql

conn = pymysql.connect(host="172.23.224.1",port=3307,user='root',password='rosemax',db='boundary_detection')
cur = conn.cursor()

#BUILD RECORDING LEVEL TABLE
cur.execute("DROP TABLE IF EXISTS TAPE") 
query = "CREATE TABLE TAPE (TAPE_ID INT AUTO_INCREMENT PRIMARY KEY, DATE_PULLED VARCHAR(20) NOT NULL, TAPE_NUMBER VARCHAR(20) NOT NULL, V_LETTER VARCHAR(1) NOT NULL, V_NUMBER VARCHAR(10) NOT NULL, FILE_NAME VARCHAR(100) NOT NULL, FILE_EXT VARCHAR(10) NOT NULL, DURATION VARCHAR(20), CC_COUNT INT, CC_CORRECT_COUNT INT)"
cur.execute(query) 

#BUILD BOUNDRY TABLE
cur.execute("DROP TABLE IF EXISTS BOUNDARY") 
query = "CREATE TABLE BOUNDARY (BOUNDARY_ID INT AUTO_INCREMENT PRIMARY KEY,TAPE_ID INT NOT NULL, BOUNDARY_START_TIME VARCHAR(20) NOT NULL, BOUNDARY_END_TIME VARCHAR(20) NOT NULL, BOUNDARY_SOURCE VARCHAR(50), CONTEXT TEXT)"
cur.execute(query)  

#BUILD CONFIDENCE TABLE
cur.execute("DROP TABLE IF EXISTS CONFIDENCE") 
query = "CREATE TABLE CONFIDENCE (CONFIDENCE_ID INT AUTO_INCREMENT PRIMARY KEY, BOUNDARY_ID VARCHAR(50) NOT NULL, CLUSTER_ID VARCHAR(20) DEFAULT 'NA', TEMPLATE_ID INT DEFAULT 0)"
cur.execute(query)   

# To commit the changes
conn.commit()         
conn.close()
