#Program for altering Column Sizes of Employee Table
#OracleAlterTableEx2.py
import cx_Oracle
def altertable():
    try:
        con=cx_Oracle.connect("system/manager@localhost/xe")
        cur=con.cursor()
        cur.execute("alter table employee add(cname varchar2(10))")
        print("Employee Table altered by adding Col Name--verify")
    except cx_Oracle.DatabaseError as kvr:
        print("Problem in Oracle DB:",kvr)

#main program
altertable()