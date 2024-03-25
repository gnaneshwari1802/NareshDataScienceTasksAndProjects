#Program inserting employee record in employee table
#OracleInsertEx1.py
import cx_Oracle
def recordinsert():
    try:
        con = cx_Oracle.connect("system/manager@localhost/xe")
        cur = con.cursor()
        iq="insert into employee values(40,'KV',0.0,'NIT')"
        cur.execute(iq)
        con.commit()
        print("Employee Record Inserted--verify")
    except cx_Oracle.DatabaseError as kvr:
        print("Problem in Oracle DB:", kvr)

#main program
recordinsert()