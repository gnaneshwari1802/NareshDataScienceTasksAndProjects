#Program for altering Column Sizes of Employee Table
#OracleAlterTableEx1.py
import cx_Oracle
def altertable():
    try:
        con=cx_Oracle.connect("system/manager@localhost/xe")
        cur=con.cursor()
        aq="alter table employee modify(eno number(3),name varchar2(15))"
        cur.execute(aq)
        print("Employee Table Column Sizes altered--verify")
    except cx_Oracle.DatabaseError as kvr:
        print("Problem in Oracle DB:",kvr)

#main program
altertable()